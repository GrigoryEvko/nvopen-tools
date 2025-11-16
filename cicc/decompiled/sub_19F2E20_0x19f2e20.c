// Function: sub_19F2E20
// Address: 0x19f2e20
//
__int64 __fastcall sub_19F2E20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r8
  int v16; // edx
  __int64 v17; // rax
  bool v18; // al
  __int64 v19; // rax
  __int64 v20; // r10
  char v21; // al
  __int64 v22; // r8
  __int64 v23; // rdx
  int v24; // r9d
  __int64 v25; // rax
  __int64 v26; // r8
  int v27; // r10d
  __int64 v28; // [rsp+10h] [rbp-90h]
  __int64 v29; // [rsp+18h] [rbp-88h]
  __int64 v30; // [rsp+18h] [rbp-88h]
  __int64 v31; // [rsp+18h] [rbp-88h]
  __int64 v32; // [rsp+20h] [rbp-80h]
  __int64 v33; // [rsp+28h] [rbp-78h]
  __int64 v34; // [rsp+28h] [rbp-78h]
  __int64 v35; // [rsp+28h] [rbp-78h]
  bool v36; // [rsp+28h] [rbp-78h]
  __int64 v37; // [rsp+28h] [rbp-78h]
  __int64 v38; // [rsp+30h] [rbp-70h] BYREF
  char v39[8]; // [rsp+38h] [rbp-68h] BYREF
  _QWORD v40[4]; // [rsp+40h] [rbp-60h] BYREF
  char v41; // [rsp+60h] [rbp-40h]

  v9 = *(unsigned int *)(a1 + 1664);
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD *)(a1 + 1648);
    v11 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v12 = (__int64 *)(v10 + 16LL * v11);
    v13 = *v12;
    if ( a2 == *v12 )
    {
LABEL_3:
      if ( v12 != (__int64 *)(v10 + 16 * v9) )
        return *((unsigned __int8 *)v12 + 8);
    }
    else
    {
      v16 = 1;
      while ( v13 != -8 )
      {
        v27 = v16 + 1;
        v11 = (v9 - 1) & (v16 + v11);
        v12 = (__int64 *)(v10 + 16LL * v11);
        v13 = *v12;
        if ( a2 == *v12 )
          goto LABEL_3;
        v16 = v27;
      }
    }
  }
  v28 = a1 + 1640;
  v33 = *(_QWORD *)(a1 + 8);
  v17 = sub_19E73A0(a1, a2);
  v18 = sub_15CC890(v33, v17, a3);
  if ( v18 )
  {
    v36 = v18;
    v38 = a2;
    v39[0] = 1;
LABEL_28:
    sub_19F2CB0((__int64)v40, v28, &v38, v39);
    return v36;
  }
  if ( *(_BYTE *)(a2 + 16) == 77 )
  {
    v36 = 0;
    if ( a3 == sub_19E73A0(a1, a2) )
    {
      v38 = a2;
      v39[0] = 0;
      goto LABEL_28;
    }
  }
  v19 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v20 = *(_QWORD *)(a2 - 8);
    v32 = v20 + v19;
  }
  else
  {
    v32 = a2;
    v20 = a2 - v19;
  }
  for ( ; v32 != v20; v20 += 24 )
  {
    v34 = *(_QWORD *)v20;
    if ( *(_BYTE *)(*(_QWORD *)v20 + 16LL) > 0x17u )
    {
      v38 = a2;
      v29 = v20;
      v21 = sub_19E9520(v28, &v38, v40);
      v20 = v29;
      v22 = v34;
      if ( !v21 || v40[0] == *(_QWORD *)(a1 + 1648) + 16LL * *(unsigned int *)(a1 + 1664) )
      {
        v23 = v34;
        v35 = v29;
        v30 = v22;
        sub_19E5420((__int64)v40, a4, v23);
        v20 = v35;
        if ( v41 )
        {
          v25 = *(unsigned int *)(a5 + 8);
          v26 = v30;
          if ( (unsigned int)v25 >= *(_DWORD *)(a5 + 12) )
          {
            v31 = v35;
            v37 = v26;
            sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, v26, v24);
            v25 = *(unsigned int *)(a5 + 8);
            v20 = v31;
            v26 = v37;
          }
          *(_QWORD *)(*(_QWORD *)a5 + 8 * v25) = v26;
          ++*(_DWORD *)(a5 + 8);
        }
      }
      else if ( !*(_BYTE *)(v40[0] + 8LL) )
      {
        v38 = a2;
        v39[0] = 0;
        sub_19F2CB0((__int64)v40, v28, &v38, v39);
        return 0;
      }
    }
  }
  return 1;
}
