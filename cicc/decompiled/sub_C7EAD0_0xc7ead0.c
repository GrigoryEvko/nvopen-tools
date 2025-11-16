// Function: sub_C7EAD0
// Address: 0xc7ead0
//
__int64 __fastcall sub_C7EAD0(__int64 a1, const char ***a2, char a3, unsigned __int8 a4, unsigned __int16 a5)
{
  __int64 *v6; // r12
  bool v8; // zf
  unsigned __int8 v9; // al
  const char *v10; // rdx
  size_t v11; // rax
  unsigned __int16 v13; // [rsp+4h] [rbp-15Ch]
  unsigned __int16 v14; // [rsp+8h] [rbp-158h]
  const char **v15; // [rsp+8h] [rbp-158h]
  const char *v16; // [rsp+10h] [rbp-150h] BYREF
  size_t v17; // [rsp+18h] [rbp-148h]
  __int64 v18; // [rsp+20h] [rbp-140h]
  _BYTE v19[312]; // [rsp+28h] [rbp-138h] BYREF

  v6 = (__int64 *)a2;
  v8 = *((_BYTE *)a2 + 33) == 1;
  v16 = v19;
  v17 = 0;
  v18 = 256;
  if ( !v8 )
    goto LABEL_6;
  v9 = *((_BYTE *)a2 + 32);
  if ( v9 == 1 )
    goto LABEL_8;
  if ( (unsigned __int8)(v9 - 3) > 3u )
  {
LABEL_6:
    a2 = (const char ***)&v16;
    v14 = a5;
    sub_CA0EC0(v6, &v16);
    v11 = v17;
    v10 = v16;
    a5 = v14;
    goto LABEL_7;
  }
  if ( v9 != 4 )
  {
    if ( v9 > 4u )
    {
      if ( (unsigned __int8)(v9 - 5) <= 1u )
      {
        v11 = (size_t)a2[1];
        v10 = (const char *)*a2;
        goto LABEL_7;
      }
LABEL_20:
      BUG();
    }
    if ( v9 != 3 )
      goto LABEL_20;
    if ( *a2 )
    {
      v13 = a5;
      v15 = *a2;
      v11 = strlen((const char *)*a2);
      v10 = (const char *)v15;
      a5 = v13;
      goto LABEL_7;
    }
LABEL_8:
    a2 = (const char ***)v6;
    sub_C7EA90(a1, v6, a3, a4, 0, a5);
    goto LABEL_9;
  }
  v10 = **a2;
  v11 = (size_t)(*a2)[1];
LABEL_7:
  if ( v11 != 1 || *v10 != 45 )
    goto LABEL_8;
  sub_C7DF90(a1);
LABEL_9:
  if ( v16 != v19 )
    _libc_free(v16, a2);
  return a1;
}
