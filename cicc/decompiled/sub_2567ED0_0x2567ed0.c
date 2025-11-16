// Function: sub_2567ED0
// Address: 0x2567ed0
//
__int64 __fastcall sub_2567ED0(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // r13d
  unsigned int v5; // r12d
  __int64 v7; // r15
  int v8; // eax
  int v9; // r9d
  unsigned int v10; // ecx
  int v11; // r11d
  __int64 *v12; // r10
  unsigned int v13; // edx
  __int64 v14; // r14
  int v15; // r8d
  unsigned int v16; // edx
  bool v17; // al
  __int64 *v18; // [rsp+8h] [rbp-98h]
  int v19; // [rsp+10h] [rbp-90h]
  unsigned int v20; // [rsp+1Ch] [rbp-84h]
  unsigned int v21; // [rsp+20h] [rbp-80h]
  int v22; // [rsp+24h] [rbp-7Ch]
  int v23; // [rsp+28h] [rbp-78h]
  int v24; // [rsp+44h] [rbp-5Ch]
  __int64 v25; // [rsp+48h] [rbp-58h]
  __int64 v26; // [rsp+50h] [rbp-50h] BYREF
  int v27; // [rsp+58h] [rbp-48h]
  __int64 v28; // [rsp+60h] [rbp-40h] BYREF
  int v29; // [rsp+68h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v27 = 0;
  v26 = -1;
  v29 = 0;
  v28 = -2;
  v8 = sub_C4F140(a2);
  v9 = v4 - 1;
  v10 = *(_DWORD *)(a2 + 8);
  v11 = 0;
  v24 = 1;
  v12 = &v28;
  v13 = (v4 - 1) & v8;
  v25 = 0;
  while ( 1 )
  {
    v14 = v7 + 16LL * v13;
    v15 = *(_DWORD *)(v14 + 8);
    if ( v15 == v10 )
    {
      if ( v10 <= 0x40 )
      {
        if ( *(_QWORD *)a2 == *(_QWORD *)v14 )
        {
LABEL_10:
          *a3 = v14;
          v5 = 1;
          goto LABEL_11;
        }
      }
      else
      {
        v21 = v13;
        v18 = v12;
        v19 = v11;
        v20 = v10;
        v22 = v9;
        v23 = *(_DWORD *)(v14 + 8);
        v17 = sub_C43C50(a2, (const void **)(v7 + 16LL * v13));
        v15 = v23;
        v9 = v22;
        v13 = v21;
        v10 = v20;
        v11 = v19;
        v12 = v18;
        if ( v17 )
          goto LABEL_10;
      }
    }
    if ( v15 == v11 )
      break;
LABEL_7:
    v16 = v24 + v13;
    ++v24;
    v13 = v9 & v16;
  }
  if ( *(_QWORD *)v14 != -1 )
  {
    if ( *(_QWORD *)v14 == -2 )
    {
      if ( v25 )
        v14 = v25;
      v25 = v14;
    }
    goto LABEL_7;
  }
  if ( v25 )
    v14 = v25;
  *a3 = v14;
  v5 = 0;
LABEL_11:
  sub_969240(v12);
  sub_969240(&v26);
  return v5;
}
