// Function: sub_2FDD600
// Address: 0x2fdd600
//
__int64 __fastcall sub_2FDD600(__int64 a1, const char *a2, __int64 a3, __int64 a4)
{
  const char *v4; // r14
  __int64 (__fastcall *v6)(__int64); // rax
  char v7; // r15
  char v8; // bl
  const char *v9; // r12
  size_t v10; // rax
  int v12; // eax
  bool v13; // zf
  _BOOL4 v14; // ebx
  int v15; // eax
  char *v16; // rdi
  unsigned __int64 v17; // rax
  __int64 v18; // rcx
  int v19; // [rsp+8h] [rbp-48h]
  unsigned int v20; // [rsp+Ch] [rbp-44h]
  char *endptr; // [rsp+18h] [rbp-38h] BYREF

  v4 = a2;
  v6 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a3 + 64LL);
  if ( v6 == sub_106E230 )
    v19 = *(_DWORD *)(a3 + 24);
  else
    v19 = ((__int64 (__fastcall *)(__int64, __int64))v6)(a3, a4);
  v7 = *a2;
  v20 = 0;
  if ( *a2 )
  {
    v8 = 1;
    while ( 1 )
    {
      if ( v7 == 10 )
        goto LABEL_6;
      v9 = *(const char **)(a3 + 40);
      v10 = strlen(v9);
      if ( !strncmp(v4, v9, v10) || (v8 &= strncmp(v4, *(const char **)(a3 + 48), *(_QWORD *)(a3 + 56)) != 0) != 0 )
      {
        if ( v7 != 32 )
        {
          v8 = 1;
          if ( (unsigned __int8)(v7 - 9) > 4u )
          {
            v12 = v19;
            v13 = memcmp(v4, ".space", 6u) == 0;
            v14 = !v13;
            if ( v13 )
            {
              v15 = strtol(v4 + 6, &endptr, 10);
              v16 = endptr;
              if ( v15 >= 0 )
                v14 = v15;
              v17 = (unsigned __int8)*endptr;
              if ( (_BYTE)v17 == 10 )
              {
LABEL_21:
                v12 = v14;
              }
              else
              {
                while ( (unsigned __int8)v17 <= 0x20u )
                {
                  v18 = 0x100003A00LL;
                  if ( !_bittest64(&v18, v17) )
                  {
                    if ( !(_BYTE)v17 )
                    {
                      v12 = v14;
                      goto LABEL_15;
                    }
                    break;
                  }
                  endptr = ++v16;
                  v17 = (unsigned __int8)*v16;
                  if ( (_BYTE)v17 == 10 )
                  {
                    v12 = v14;
                    goto LABEL_15;
                  }
                }
                if ( !strncmp(v16, *(const char **)(a3 + 48), *(_QWORD *)(a3 + 56)) )
                  goto LABEL_21;
                v12 = v19;
              }
            }
LABEL_15:
            v20 += v12;
            v8 = 0;
          }
          goto LABEL_7;
        }
LABEL_6:
        v8 = 1;
LABEL_7:
        v7 = *++v4;
        if ( !v7 )
          return v20;
      }
      else
      {
        v7 = *++v4;
        if ( !v7 )
          return v20;
      }
    }
  }
  return v20;
}
