// Function: sub_22CCF60
// Address: 0x22ccf60
//
__int64 __fastcall sub_22CCF60(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v7; // al
  unsigned __int8 v9; // al
  int v10; // eax
  int v11; // eax
  char v14; // al
  bool v15; // zf
  unsigned __int64 v16; // rax
  int v17; // edx
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // [rsp+20h] [rbp-C0h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-B8h]
  __int64 v23; // [rsp+50h] [rbp-90h] BYREF
  __int64 v24; // [rsp+58h] [rbp-88h] BYREF
  __int64 v25[3]; // [rsp+60h] [rbp-80h] BYREF
  char v26; // [rsp+78h] [rbp-68h]
  __int64 v27; // [rsp+80h] [rbp-60h] BYREF
  __int64 v28; // [rsp+88h] [rbp-58h]
  int v29; // [rsp+90h] [rbp-50h]
  __int64 v30; // [rsp+98h] [rbp-48h]
  int v31; // [rsp+A0h] [rbp-40h]
  char v32; // [rsp+A8h] [rbp-38h]

  v7 = *(_BYTE *)a3;
  if ( *(_BYTE *)a3 <= 0x15u )
  {
    LOWORD(v27) = 0;
    if ( (unsigned __int8)(v7 - 12) <= 1u )
    {
      *(_WORD *)a1 = 1;
      goto LABEL_4;
    }
    if ( v7 == 17 )
    {
      v22 = *(_DWORD *)(a3 + 32);
      if ( v22 > 0x40 )
        sub_C43780((__int64)&v21, (const void **)(a3 + 24));
      else
        v21 = *(_QWORD *)(a3 + 24);
      sub_AADBC0((__int64)&v23, &v21);
      sub_22C00F0((__int64)&v27, (__int64)&v23, 0, 0, 1u);
      sub_969240(v25);
      sub_969240(&v23);
      sub_969240(&v21);
      v9 = v27;
      *(_WORD *)a1 = (unsigned __int8)v27;
      if ( v9 > 3u )
      {
        if ( (unsigned __int8)(v9 - 4) <= 1u )
        {
          v10 = v29;
          v29 = 0;
          *(_DWORD *)(a1 + 16) = v10;
          *(_QWORD *)(a1 + 8) = v28;
          v11 = v31;
          v31 = 0;
          *(_DWORD *)(a1 + 32) = v11;
          *(_QWORD *)(a1 + 24) = v30;
          *(_BYTE *)(a1 + 1) = BYTE1(v27);
        }
        goto LABEL_4;
      }
      if ( v9 <= 1u )
      {
LABEL_4:
        *(_BYTE *)(a1 + 40) = 1;
        LOBYTE(v27) = 0;
        sub_22C0090((unsigned __int8 *)&v27);
        return a1;
      }
    }
    else
    {
      v28 = a3;
      *(_WORD *)a1 = 2;
    }
    *(_QWORD *)(a1 + 8) = v28;
    goto LABEL_4;
  }
  sub_22CB610((__int64)&v23, a2, (__int64 *)a3, a4, a5, 1u);
  if ( !v26 )
  {
    *(_BYTE *)(a1 + 40) = 0;
    return a1;
  }
  v14 = v23;
  if ( (unsigned __int8)(v23 - 4) <= 1u )
  {
    if ( sub_9876C0(&v24) )
    {
LABEL_18:
      v15 = v26 == 0;
      *(_BYTE *)(a1 + 40) = 0;
      if ( v15 )
        return a1;
      sub_22C0650(a1, (unsigned __int8 *)&v23);
      *(_BYTE *)(a1 + 40) = 1;
      goto LABEL_30;
    }
    v14 = v23;
  }
  if ( v14 == 2 )
    goto LABEL_18;
  v16 = *(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v16 == a4 + 48 )
  {
    v18 = 0;
  }
  else
  {
    if ( !v16 )
      BUG();
    v17 = *(unsigned __int8 *)(v16 - 24);
    v18 = 0;
    v19 = v16 - 24;
    if ( (unsigned int)(v17 - 30) < 0xB )
      v18 = v19;
  }
  sub_22C7100((__int64)&v27, a2, a3, a4, v18);
  if ( v32 )
  {
    sub_22C6BD0(a2, a3, (unsigned __int8 *)&v27, a6, v20, &v27);
    sub_22EACA0(&v21, &v23, &v27);
    sub_22C0650(a1, (unsigned __int8 *)&v21);
    *(_BYTE *)(a1 + 40) = 1;
    sub_22C0090((unsigned __int8 *)&v21);
    if ( v32 )
    {
      v32 = 0;
      sub_22C0090((unsigned __int8 *)&v27);
    }
  }
  else
  {
    *(_BYTE *)(a1 + 40) = 0;
  }
LABEL_30:
  if ( v26 )
  {
    v26 = 0;
    sub_22C0090((unsigned __int8 *)&v23);
  }
  return a1;
}
