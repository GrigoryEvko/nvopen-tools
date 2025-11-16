// Function: sub_E0D760
// Address: 0xe0d760
//
void __fastcall sub_E0D760(__int64 a1, void **a2, unsigned __int64 *a3)
{
  unsigned __int64 v4; // rdx
  _BYTE *v5; // rbx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // rdx
  const char *v12; // rbx
  size_t v13; // rax
  unsigned __int64 v14; // rax
  char *v15; // rsi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // [rsp+8h] [rbp-48h] BYREF
  size_t v18; // [rsp+10h] [rbp-40h] BYREF
  const char *v19; // [rsp+18h] [rbp-38h]

  v4 = *a3;
  if ( !v4 )
    goto LABEL_16;
  v5 = (_BYTE *)a3[1];
  if ( *v5 != 81 )
  {
    if ( (unsigned int)((char)*v5 - 48) <= 9 )
    {
      sub_E0D090(a3, &v18);
      v8 = *a3;
      if ( !*a3 )
      {
LABEL_16:
        a3[1] = 0;
        return;
      }
      v9 = v18;
      if ( v18 <= v8 && v18 )
      {
        if ( v18 > 3 && v8 > 2 )
        {
          v10 = a3[1];
          if ( *(_WORD *)v10 == 24415 && *(_BYTE *)(v10 + 2) == 83 )
          {
            v14 = v8 - 3;
            v15 = (char *)(v10 + 3);
            v16 = *a3 - v18;
            if ( v16 < v14 )
            {
              while ( (unsigned int)(*v15 - 48) <= 9 )
              {
                --v14;
                ++v15;
                if ( v14 <= v16 )
                  goto LABEL_28;
              }
            }
            else
            {
LABEL_28:
              if ( v14 == v16 )
              {
                *a3 = v14;
                a3[1] = v10 + v9;
                sub_E0D760(a1, a2, a3);
                return;
              }
            }
          }
        }
        sub_E0D290(a2, a3, v18);
        return;
      }
    }
LABEL_15:
    *a3 = 0;
    goto LABEL_16;
  }
  v11 = v4 - 1;
  v18 = 0;
  v19 = 0;
  a3[1] = (unsigned __int64)(v5 + 1);
  *a3 = v11;
  if ( !v11 )
    goto LABEL_15;
  if ( !(unsigned __int8)sub_E0CFB0((__int64 *)a3, (__int64 *)&v17) )
    goto LABEL_15;
  if ( (__int64)&v5[-*(_QWORD *)(a1 + 8)] < (__int64)v17 )
    goto LABEL_15;
  v12 = &v5[-v17];
  v13 = strlen(v12);
  v19 = v12;
  v18 = v13;
  if ( !v13 )
    goto LABEL_15;
  if ( (unsigned int)(*v12 - 48) > 9 )
    goto LABEL_15;
  sub_E0D090(&v18, &v17);
  if ( !v18 )
    goto LABEL_15;
  if ( v18 < v17 )
    goto LABEL_15;
  sub_E0D290(a2, &v18, v17);
  if ( !v18 )
    goto LABEL_15;
}
