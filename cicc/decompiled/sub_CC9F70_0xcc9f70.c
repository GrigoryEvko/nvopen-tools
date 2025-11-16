// Function: sub_CC9F70
// Address: 0xcc9f70
//
__int64 __fastcall sub_CC9F70(__int64 a1, void **a2)
{
  const char *v2; // rsi
  char *v3; // rax
  char *v4; // rax
  __int64 v5; // r9
  __int64 result; // rax
  const void **v7; // r12
  int v8; // eax
  unsigned int v9; // edx
  int v10; // edx
  unsigned __int64 v11; // r14
  _WORD *v12; // r15
  int v13; // eax
  int v14; // edx
  char *v15[2]; // [rsp+0h] [rbp-90h] BYREF
  const void **v16; // [rsp+10h] [rbp-80h] BYREF
  __int64 v17; // [rsp+18h] [rbp-78h]
  _BYTE v18[112]; // [rsp+20h] [rbp-70h] BYREF

  sub_CA0F50((__int64 *)a1, a2);
  v2 = (const char *)&v16;
  v17 = 0x400000000LL;
  v3 = *(char **)a1;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  v15[0] = v3;
  v4 = *(char **)(a1 + 8);
  v16 = (const void **)v18;
  v15[1] = v4;
  sub_C93960(v15, (__int64)&v16, 45, 3, 1, v5);
  if ( !(_DWORD)v17 )
  {
    result = *(unsigned int *)(a1 + 52);
    v7 = v16;
    goto LABEL_3;
  }
  *(_DWORD *)(a1 + 32) = sub_CC8470(*v16, (unsigned __int64)v16[1]);
  v2 = (const char *)v16[1];
  v8 = sub_CC5470(*v16, (unsigned __int64)v2);
  v9 = v17;
  v7 = v16;
  *(_DWORD *)(a1 + 36) = v8;
  if ( v9 <= 1 )
  {
    v11 = (unsigned __int64)v7[1];
    v12 = *v7;
    if ( v11 <= 6 )
    {
      if ( v11 == 6 )
      {
        if ( *(_DWORD *)v12 == 1936746861 && v12[2] == 13366 )
          goto LABEL_38;
        if ( *(_DWORD *)v12 == 1936746861 && v12[2] == 27749 )
        {
LABEL_18:
          v13 = 1;
LABEL_19:
          *(_DWORD *)(a1 + 48) = v13;
          goto LABEL_9;
        }
        if ( *(_DWORD *)v12 == 1936746861 )
        {
          if ( v12[2] != 13938 )
          {
            v13 = 0;
            goto LABEL_19;
          }
          goto LABEL_18;
        }
LABEL_24:
        v13 = 0;
        goto LABEL_19;
      }
      if ( v11 == 4 )
      {
        if ( *(_DWORD *)v12 != 1936746861 )
        {
          v13 = 0;
          goto LABEL_19;
        }
        goto LABEL_18;
      }
    }
    else
    {
      if ( *(_DWORD *)v12 == 1936746861 && v12[2] == 13166 )
      {
        v13 = 3;
        if ( *((_BYTE *)v12 + 6) == 50 )
          goto LABEL_19;
      }
      v2 = "mips64";
      if ( !memcmp(*v7, "mips64", 6u) )
        goto LABEL_38;
      if ( v11 > 8 )
      {
        if ( *(_QWORD *)v12 == 0x366173697370696DLL && *((_BYTE *)v12 + 8) == 52 )
        {
LABEL_38:
          v13 = 4;
          goto LABEL_19;
        }
        if ( *(_QWORD *)v12 == 0x336173697370696DLL && *((_BYTE *)v12 + 8) == 50 )
          goto LABEL_18;
      }
    }
    if ( v11 == 8 )
    {
      v13 = *(_QWORD *)v12 == 0x6C6536727370696DLL;
      goto LABEL_19;
    }
    goto LABEL_24;
  }
  v2 = (const char *)v7[3];
  *(_DWORD *)(a1 + 40) = sub_CC4230((__int64)v7[2], (__int64)v2);
  if ( v10 == 2
    || (v2 = (const char *)v7[5], *(_DWORD *)(a1 + 44) = sub_CC4400((__int64)v7[4], (unsigned __int64)v2), v14 == 3) )
  {
LABEL_9:
    result = *(unsigned int *)(a1 + 52);
    if ( (_DWORD)result )
      goto LABEL_4;
    goto LABEL_10;
  }
  *(_DWORD *)(a1 + 48) = sub_CC4B20((__int64)v7[6], (unsigned __int64)v7[7]);
  v2 = (const char *)v7[7];
  result = sub_CC4070((__int64)v7[6], (unsigned __int64)v2);
  *(_DWORD *)(a1 + 52) = result;
LABEL_3:
  if ( (_DWORD)result )
    goto LABEL_4;
LABEL_10:
  result = sub_CC3FA0(a1);
  *(_DWORD *)(a1 + 52) = result;
LABEL_4:
  if ( v7 != (const void **)v18 )
    return _libc_free(v7, v2);
  return result;
}
