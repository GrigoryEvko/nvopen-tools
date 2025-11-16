// Function: sub_2A625F0
// Address: 0x2a625f0
//
bool __fastcall sub_2A625F0(__int64 a1, __int64 a2, __int64 a3, char a4, unsigned int a5)
{
  unsigned __int8 v5; // al
  unsigned __int8 v6; // dl
  char v9; // dl
  unsigned int v10; // eax
  char v11; // r11
  __int64 v13; // rsi
  __int64 v14; // rdx
  char v15; // dl
  unsigned int v16; // eax
  unsigned int v17; // [rsp-9Ch] [rbp-9Ch]
  unsigned int v18; // [rsp-9Ch] [rbp-9Ch]
  bool v19; // [rsp-9Ch] [rbp-9Ch]
  unsigned int v20; // [rsp-9Ch] [rbp-9Ch]
  unsigned int v21; // [rsp-9Ch] [rbp-9Ch]
  __int64 v22; // [rsp-98h] [rbp-98h] BYREF
  unsigned int v23; // [rsp-90h] [rbp-90h]
  __int64 v24; // [rsp-88h] [rbp-88h] BYREF
  __int64 v25; // [rsp-78h] [rbp-78h] BYREF
  unsigned int v26; // [rsp-70h] [rbp-70h]
  __int64 v27; // [rsp-68h] [rbp-68h] BYREF
  unsigned int v28; // [rsp-60h] [rbp-60h]
  __int64 v29; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v30; // [rsp-50h] [rbp-50h]
  __int64 v31; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v32; // [rsp-40h] [rbp-40h]

  v5 = *(_BYTE *)a2;
  if ( !*(_BYTE *)a2 )
    return 0;
  v6 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 6 )
    return 0;
  if ( v5 == 6 )
    goto LABEL_21;
  if ( v6 != 1 )
  {
    if ( !v6 )
    {
      sub_22C0090((unsigned __int8 *)a1);
      sub_22C05A0(a1, (unsigned __int8 *)a2);
      return 1;
    }
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v5 == 1 )
        {
          *(_BYTE *)a1 = 5;
          return v6 != 5;
        }
        else
        {
          v17 = a5;
          sub_22C0B50((__int64)&v29, (char *)a2, *(_DWORD *)(a1 + 16), 1);
          sub_AB3510((__int64)&v25, a1 + 8, (__int64)&v29, 0);
          sub_969240(&v31);
          sub_969240(&v29);
          v9 = *(_BYTE *)a2 == 5;
          v30 = v26;
          v26 = 0;
          v29 = v25;
          v10 = v28;
          v28 = 0;
          v32 = v10;
          v31 = v27;
          LOBYTE(v17) = sub_2A62120((char *)a1, (__int64)&v29, v9, a4, v17);
          sub_969240(&v31);
          sub_969240(&v29);
          sub_969240(&v27);
          sub_969240(&v25);
          return v17;
        }
      }
      if ( v5 == 3 )
      {
        v11 = 0;
        if ( *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 + 8) )
          return v11;
      }
      goto LABEL_21;
    }
    if ( v5 != 2 )
    {
      v11 = 0;
      if ( v5 == 1 )
        return v11;
      v13 = *(_QWORD *)(a1 + 8);
      goto LABEL_15;
    }
    v13 = *(_QWORD *)(a1 + 8);
    if ( v13 != *(_QWORD *)(a2 + 8) )
    {
LABEL_15:
      v14 = *(_QWORD *)(v13 + 8);
      if ( (unsigned __int8)(*(_BYTE *)(v14 + 8) - 17) <= 1u && *(_BYTE *)(**(_QWORD **)(v14 + 16) + 8LL) == 12 )
      {
        v18 = a5;
        sub_AD8380((__int64)&v22, v13);
        sub_22C0B50((__int64)&v29, (char *)a2, v23, 1);
        sub_AB3510((__int64)&v25, (__int64)&v22, (__int64)&v29, 0);
        sub_969240(&v31);
        sub_969240(&v29);
        v15 = *(_BYTE *)a2 == 5;
        v30 = v26;
        v26 = 0;
        v29 = v25;
        v16 = v28;
        v28 = 0;
        v32 = v16;
        v31 = v27;
        LOBYTE(v18) = sub_2A62120((char *)a1, (__int64)&v29, v15, a4, v18);
        sub_969240(&v31);
        sub_969240(&v29);
        sub_969240(&v27);
        sub_969240(&v25);
        sub_969240(&v24);
        sub_969240(&v22);
        return v18;
      }
LABEL_21:
      sub_22C0090((unsigned __int8 *)a1);
      *(_BYTE *)a1 = 6;
      return 1;
    }
    return 0;
  }
  if ( v5 == 1 )
    return 0;
  if ( v5 != 2 )
  {
    if ( (unsigned __int8)(v5 - 4) <= 1u )
    {
      v30 = *(_DWORD *)(a2 + 16);
      if ( v30 > 0x40 )
      {
        v21 = a5;
        sub_C43780((__int64)&v29, (const void **)(a2 + 8));
        a5 = v21;
      }
      else
      {
        v29 = *(_QWORD *)(a2 + 8);
      }
      v32 = *(_DWORD *)(a2 + 32);
      if ( v32 > 0x40 )
      {
        v20 = a5;
        sub_C43780((__int64)&v31, (const void **)(a2 + 24));
        a5 = v20;
      }
      else
      {
        v31 = *(_QWORD *)(a2 + 24);
      }
      v19 = sub_2A62120((char *)a1, (__int64)&v29, 1, a4, a5);
      sub_969240(&v31);
      sub_969240(&v29);
      return v19;
    }
    goto LABEL_21;
  }
  return sub_2A624B0(a1, *(unsigned __int8 **)(a2 + 8), 1);
}
