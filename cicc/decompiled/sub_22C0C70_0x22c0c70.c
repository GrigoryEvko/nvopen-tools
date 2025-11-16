// Function: sub_22C0C70
// Address: 0x22c0c70
//
void __fastcall sub_22C0C70(__int64 a1, __int64 a2, __int64 a3, char a4, unsigned int a5)
{
  unsigned __int8 v5; // al
  unsigned __int8 v6; // dl
  char v9; // dl
  unsigned int v10; // eax
  __int64 v11; // rsi
  __int64 v12; // rdx
  char v13; // dl
  unsigned int v14; // eax
  unsigned int v15; // [rsp-ACh] [rbp-ACh]
  unsigned int v16; // [rsp-A8h] [rbp-A8h]
  unsigned int v17; // [rsp-A0h] [rbp-A0h]
  unsigned int v18; // [rsp-A0h] [rbp-A0h]
  __int64 v19; // [rsp-98h] [rbp-98h] BYREF
  unsigned int v20; // [rsp-90h] [rbp-90h]
  __int64 v21; // [rsp-88h] [rbp-88h] BYREF
  __int64 v22; // [rsp-78h] [rbp-78h] BYREF
  unsigned int v23; // [rsp-70h] [rbp-70h]
  __int64 v24; // [rsp-68h] [rbp-68h] BYREF
  unsigned int v25; // [rsp-60h] [rbp-60h]
  __int64 v26; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v27; // [rsp-50h] [rbp-50h]
  __int64 v28; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v29; // [rsp-40h] [rbp-40h]

  v5 = *(_BYTE *)a2;
  if ( !*(_BYTE *)a2 )
    return;
  v6 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 6 )
    return;
  if ( v5 == 6 )
    goto LABEL_21;
  switch ( v6 )
  {
    case 1u:
      if ( v5 == 1 )
        return;
      if ( v5 == 2 )
      {
        sub_22C0310(a1, *(unsigned __int8 **)(a2 + 8), 1);
        return;
      }
      if ( (unsigned __int8)(v5 - 4) <= 1u )
      {
        v27 = *(_DWORD *)(a2 + 16);
        if ( v27 > 0x40 )
        {
          v18 = a5;
          sub_C43780((__int64)&v26, (const void **)(a2 + 8));
          a5 = v18;
        }
        else
        {
          v26 = *(_QWORD *)(a2 + 8);
        }
        v29 = *(_DWORD *)(a2 + 32);
        if ( v29 > 0x40 )
        {
          v17 = a5;
          sub_C43780((__int64)&v28, (const void **)(a2 + 24));
          a5 = v17;
        }
        else
        {
          v28 = *(_QWORD *)(a2 + 24);
        }
        sub_22C00F0(a1, (__int64)&v26, 1, a4, a5);
        sub_969240(&v28);
        sub_969240(&v26);
        return;
      }
      goto LABEL_21;
    case 0u:
      sub_22C0090((unsigned __int8 *)a1);
      sub_22C05A0(a1, (unsigned __int8 *)a2);
      return;
    case 2u:
      if ( v5 == 2 )
      {
        v11 = *(_QWORD *)(a1 + 8);
        if ( v11 == *(_QWORD *)(a2 + 8) )
          return;
      }
      else
      {
        if ( v5 == 1 )
          return;
        v11 = *(_QWORD *)(a1 + 8);
      }
      v12 = *(_QWORD *)(v11 + 8);
      if ( (unsigned __int8)(*(_BYTE *)(v12 + 8) - 17) <= 1u && *(_BYTE *)(**(_QWORD **)(v12 + 16) + 8LL) == 12 )
      {
        v15 = a5;
        sub_AD8380((__int64)&v19, v11);
        sub_22C0B50((__int64)&v26, (char *)a2, v20, 1);
        sub_AB3510((__int64)&v22, (__int64)&v19, (__int64)&v26, 0);
        sub_969240(&v28);
        sub_969240(&v26);
        v13 = *(_BYTE *)a2 == 5;
        v27 = v23;
        v23 = 0;
        v26 = v22;
        v14 = v25;
        v25 = 0;
        v29 = v14;
        v28 = v24;
        sub_22C00F0(a1, (__int64)&v26, v13, a4, v15);
        sub_969240(&v28);
        sub_969240(&v26);
        sub_969240(&v24);
        sub_969240(&v22);
        sub_969240(&v21);
        sub_969240(&v19);
        return;
      }
      goto LABEL_21;
  }
  if ( v6 != 3 )
  {
    if ( v5 == 1 )
    {
      *(_BYTE *)a1 = 5;
    }
    else
    {
      v16 = a5;
      sub_22C0B50((__int64)&v26, (char *)a2, *(_DWORD *)(a1 + 16), 1);
      sub_AB3510((__int64)&v22, a1 + 8, (__int64)&v26, 0);
      sub_969240(&v28);
      sub_969240(&v26);
      v9 = *(_BYTE *)a2 == 5;
      v27 = v23;
      v23 = 0;
      v26 = v22;
      v10 = v25;
      v25 = 0;
      v29 = v10;
      v28 = v24;
      sub_22C00F0(a1, (__int64)&v26, v9, a4, v16);
      sub_969240(&v28);
      sub_969240(&v26);
      sub_969240(&v24);
      sub_969240(&v22);
    }
    return;
  }
  if ( v5 != 3 || *(_QWORD *)(a1 + 8) != *(_QWORD *)(a2 + 8) )
  {
LABEL_21:
    sub_22C0090((unsigned __int8 *)a1);
    *(_BYTE *)a1 = 6;
  }
}
