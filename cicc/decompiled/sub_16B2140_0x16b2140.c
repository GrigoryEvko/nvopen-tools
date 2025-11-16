// Function: sub_16B2140
// Address: 0x16b2140
//
__int64 __fastcall sub_16B2140(
        const char *a1,
        __int64 a2,
        __int64 a3,
        char *a4,
        __int64 a5,
        int a6,
        const char *a7,
        _DWORD *a8)
{
  __int64 v10; // r12
  unsigned __int8 v11; // al
  int v12; // ebx
  int v13; // eax
  __int64 v14; // rsi
  __int64 v15; // rdx
  char *v16; // rcx
  size_t v17; // rax
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // r8
  const char *v22; // rax
  char *v23; // rdx
  size_t v24; // rax
  unsigned __int8 i; // [rsp-10h] [rbp-B0h]
  char v27; // [rsp+13h] [rbp-8Dh]
  const char *v28; // [rsp+18h] [rbp-88h]
  char *v29; // [rsp+20h] [rbp-80h] BYREF
  __int64 v30; // [rsp+28h] [rbp-78h]
  _QWORD v31[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v32; // [rsp+40h] [rbp-60h]
  _QWORD v33[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v34; // [rsp+60h] [rbp-40h]

  v10 = (__int64)a1;
  v11 = a1[12];
  v29 = a4;
  v30 = a5;
  v12 = *((_DWORD *)a1 + 5);
  if ( (v11 & 0x18) != 0 )
  {
    v13 = (v11 >> 3) & 3;
    if ( v13 != 2 )
    {
LABEL_3:
      if ( v13 != 3 )
      {
        LODWORD(v14) = *a8;
        goto LABEL_5;
      }
      if ( !v12 )
      {
        if ( v29 )
        {
          v19 = sub_16E8CB0(a1, a2, a3);
          v20 = 1283;
          v21 = v19;
          v32 = 1283;
          v31[0] = "does not allow a value! '";
          v31[1] = &v29;
          v33[0] = v31;
          v33[1] = "' specified.";
          v34 = 770;
          return sub_16B1F90(v10, (__int64)v33, 0, 0, v21, v20);
        }
        LODWORD(v14) = *a8;
        return sub_16B01B0(v10, v14, a2, a3, v29, v30, 0);
      }
      HIBYTE(v34) = 1;
      v21 = sub_16E8CB0(a1, a2, a3);
      v22 = "multi-valued option specified with ValueDisallowed modifier!";
LABEL_28:
      v33[0] = v22;
      LOBYTE(v34) = 3;
      return sub_16B1F90(v10, (__int64)v33, 0, 0, v21, v20);
    }
  }
  else
  {
    v13 = (*(__int64 (__fastcall **)(const char *))(*(_QWORD *)a1 + 8LL))(a1);
    if ( v13 != 2 )
      goto LABEL_3;
  }
  LODWORD(v14) = *a8;
  if ( v29 )
  {
    if ( v12 )
      goto LABEL_19;
    return sub_16B01B0(v10, v14, a2, a3, v29, v30, 0);
  }
  v14 = (unsigned int)(v14 + 1);
  if ( (int)v14 >= a6 )
  {
    HIBYTE(v34) = 1;
    v21 = sub_16E8CB0(a1, v14, a3);
    v22 = "requires a value!";
    goto LABEL_28;
  }
  a1 = a7;
  *a8 = v14;
  v23 = *(char **)&a7[8 * (int)v14];
  v24 = 0;
  if ( v23 )
  {
    a1 = *(const char **)&a7[8 * (int)v14];
    v24 = strlen(a1);
    v23 = (char *)a1;
  }
  v29 = v23;
  v30 = v24;
LABEL_5:
  if ( v12 )
  {
    v15 = 0;
    if ( !v29 )
      goto LABEL_10;
LABEL_19:
    for ( i = 0; ; i = v15 )
    {
      a1 = (const char *)v10;
      if ( (unsigned __int8)sub_16B01B0(v10, v14, a2, a3, v29, v30, i) )
        return 1;
      if ( !--v12 )
        return 0;
      LODWORD(v14) = *a8;
      v15 = 1;
LABEL_10:
      v14 = (unsigned int)(v14 + 1);
      if ( (int)v14 >= a6 )
        break;
      *a8 = v14;
      v16 = *(char **)&a7[8 * (int)v14];
      v17 = 0;
      if ( v16 )
      {
        v27 = v15;
        v28 = *(const char **)&a7[8 * (int)v14];
        v17 = strlen(v28);
        LOBYTE(v15) = v27;
        v16 = (char *)v28;
      }
      v29 = v16;
      v30 = v17;
    }
    HIBYTE(v34) = 1;
    v21 = sub_16E8CB0(a1, v14, v15);
    v22 = "not enough values!";
    goto LABEL_28;
  }
  return sub_16B01B0(v10, v14, a2, a3, v29, v30, 0);
}
