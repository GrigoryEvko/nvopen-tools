// Function: sub_1DD9760
// Address: 0x1dd9760
//
__int64 __fastcall sub_1DD9760(__int64 a1, unsigned __int16 a2, __int64 a3)
{
  unsigned int v3; // r14d
  bool v6; // al
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r15
  __int64 (*v16)(); // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned int v19; // eax
  __int64 v20; // r13
  __int64 v21; // rsi
  __int64 v22; // r15
  __int64 v23; // rdx
  __int64 v24; // rax
  char *v26; // rsi
  __int64 v28; // [rsp+10h] [rbp-80h]
  unsigned int v29; // [rsp+18h] [rbp-78h]
  bool v30; // [rsp+1Fh] [rbp-71h]
  __int64 v31; // [rsp+28h] [rbp-68h] BYREF
  __int64 v32; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v33; // [rsp+38h] [rbp-58h]
  __int64 v34; // [rsp+40h] [rbp-50h]
  __int64 v35; // [rsp+48h] [rbp-48h]
  __int64 v36; // [rsp+50h] [rbp-40h]

  v3 = a2;
  v6 = sub_1DD6670(a1, a2, -1);
  v7 = *(_QWORD *)(a1 + 32);
  v30 = v6;
  v8 = sub_1DD5D40(a1, v7);
  v11 = 0;
  v12 = v8;
  v13 = *(_QWORD *)(a1 + 56);
  v14 = *(_QWORD *)(v13 + 16);
  v15 = *(_QWORD *)(v13 + 40);
  v16 = *(__int64 (**)())(*(_QWORD *)v14 + 40LL);
  if ( v16 != sub_1D00B00 )
    v11 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, _QWORD))v16)(v14, v7, v9, v10, 0);
  if ( v30 )
  {
    v17 = a1 + 24;
    if ( a1 + 24 != v12 )
    {
      while ( **(_WORD **)(v12 + 16) == 15 )
      {
        v18 = *(_QWORD *)(v12 + 32);
        if ( *(_DWORD *)(v18 + 48) == v3 )
        {
          v29 = *(_DWORD *)(v18 + 8);
          sub_1E69410(v15, v29, a3, 0);
          return v29;
        }
        if ( (*(_BYTE *)v12 & 4) != 0 )
        {
          v12 = *(_QWORD *)(v12 + 8);
          if ( v17 == v12 )
            break;
        }
        else
        {
          while ( (*(_BYTE *)(v12 + 46) & 8) != 0 )
            v12 = *(_QWORD *)(v12 + 8);
          v12 = *(_QWORD *)(v12 + 8);
          if ( v17 == v12 )
            break;
        }
      }
    }
  }
  v28 = v11;
  v19 = sub_1E6B9A0(v15, a3, byte_3F871B3, 0);
  v20 = *(_QWORD *)(a1 + 56);
  v29 = v19;
  v21 = *(_QWORD *)(v28 + 8);
  v31 = 0;
  v22 = sub_1E0B640(v20, v21 + 960, &v31, 0, v28);
  sub_1DD5BA0((__int64 *)(a1 + 16), v22);
  v23 = *(_QWORD *)v12;
  v24 = *(_QWORD *)v22;
  *(_QWORD *)(v22 + 8) = v12;
  v23 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v22 = v23 | v24 & 7;
  *(_QWORD *)(v23 + 8) = v22;
  *(_QWORD *)v12 = v22 | *(_QWORD *)v12 & 7LL;
  v32 = 0x10000000;
  v33 = v29;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  sub_1E1A9C0(v22, v20, &v32);
  v32 = 0x40000000;
  v34 = 0;
  v33 = v3;
  v35 = 0;
  v36 = 0;
  sub_1E1A9C0(v22, v20, &v32);
  if ( v31 )
    sub_161E7C0((__int64)&v31, v31);
  if ( !v30 )
  {
    HIDWORD(v32) = -1;
    v26 = *(char **)(a1 + 160);
    LOWORD(v32) = a2;
    if ( v26 == *(char **)(a1 + 168) )
    {
      sub_1D4B220((char **)(a1 + 152), v26, &v32);
    }
    else
    {
      if ( v26 )
      {
        *(_QWORD *)v26 = v32;
        v26 = *(char **)(a1 + 160);
      }
      *(_QWORD *)(a1 + 160) = v26 + 8;
    }
  }
  return v29;
}
