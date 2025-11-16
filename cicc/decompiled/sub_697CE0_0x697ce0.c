// Function: sub_697CE0
// Address: 0x697ce0
//
__int64 __fastcall sub_697CE0(__int64 a1, _QWORD *a2, __int64 *a3, _DWORD *a4)
{
  _BYTE *v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdi
  int v15; // r12d
  unsigned __int64 v16; // rsi
  _BOOL4 v17; // edx
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r12
  char v23; // al
  unsigned int v25; // eax
  FILE *v26; // rsi
  _BOOL4 v27; // [rsp+Ch] [rbp-E4h]
  int v28; // [rsp+14h] [rbp-DCh] BYREF
  unsigned int v29; // [rsp+18h] [rbp-D8h] BYREF
  char v30[4]; // [rsp+1Ch] [rbp-D4h] BYREF
  __int64 v31; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v32; // [rsp+28h] [rbp-C8h] BYREF
  _BYTE v33[192]; // [rsp+30h] [rbp-C0h] BYREF

  v32 = 0;
  sub_6E1DD0(&v31);
  v8 = v33;
  v9 = 4;
  sub_6E1E00(4, v33, 0, 1);
  if ( (*(_BYTE *)(a1 + 180) & 2) == 0 )
  {
    v14 = *a3;
    v15 = 0;
    if ( (*(_BYTE *)(v14 + 140) & 0xFB) == 8 )
      v15 = sub_8D4C10(v14, dword_4F077C4 != 2);
    v16 = 0;
    v17 = (*((_BYTE *)a2 + 25) & 1) == 0;
    if ( (*(_BYTE *)(*a2 + 140LL) & 0xFB) == 8 )
    {
      v27 = (*((_BYTE *)a2 + 25) & 1) == 0;
      v25 = sub_8D4C10(*a2, dword_4F077C4 != 2);
      v17 = v27;
      v16 = v25;
    }
    v18 = a1;
    v22 = sub_841B50(a1, v16, v17, v15, (_DWORD)a4, (unsigned int)&v28, (__int64)&v29, (__int64)&v32, (__int64)v30);
    if ( !v29 )
    {
      if ( !v28 )
      {
        if ( v22 )
        {
          sub_6E6130(v22, a4, 0, 0);
          sub_6E2B30(v22, a4);
          sub_6E1DF0(v31);
          goto LABEL_8;
        }
        if ( (unsigned int)sub_6E5430(a1, v16, v29, v19, v20, v21) )
        {
          if ( (*(_BYTE *)(*a2 + 140LL) & 0xFB) != 8 || (unsigned int)sub_8D4C10(*a2, dword_4F077C4 != 2) != 1 || v32 )
          {
            v26 = (FILE *)sub_67DA80(0x174u, a4, a1);
            sub_87CA90(v32, v26);
            sub_685910((__int64)v26, v26);
            sub_6E2B30(v26, v26);
            sub_6E1DF0(v31);
            return v22;
          }
          v16 = (unsigned __int64)a4;
          v18 = 371;
          sub_685360(0x173u, a4, a1);
        }
        sub_6E2B30(v18, v16);
        sub_6E1DF0(v31);
        return v22;
      }
      if ( (unsigned int)sub_6E5430(a1, v16, v29, v19, v20, v21) )
      {
        v16 = (unsigned __int64)a4;
        v18 = 373;
        sub_685360(0x175u, a4, a1);
      }
    }
    sub_6E2B30(v18, v16);
    sub_6E1DF0(v31);
    if ( v22 )
    {
LABEL_8:
      v23 = *(_BYTE *)(v22 + 80);
      if ( v23 == 16 )
      {
        v22 = **(_QWORD **)(v22 + 88);
        v23 = *(_BYTE *)(v22 + 80);
      }
      if ( v23 == 24 )
        v22 = *(_QWORD *)(v22 + 88);
      return *(_QWORD *)(v22 + 88);
    }
    return v22;
  }
  if ( (unsigned int)sub_6E5430(4, v33, v10, v11, v12, v13) )
  {
    v8 = a4;
    v9 = 372;
    sub_685360(0x174u, a4, a1);
  }
  sub_6E2B30(v9, v8);
  sub_6E1DF0(v31);
  return 0;
}
