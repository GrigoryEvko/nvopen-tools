// Function: sub_CB6320
// Address: 0xcb6320
//
void *__fastcall sub_CB6320(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  void *result; // rax
  size_t v6; // rdx
  unsigned __int64 v7; // r13
  int v8; // eax
  unsigned int v9; // r13d
  int v10; // edx
  char v11; // si
  char *v12; // rax
  unsigned __int8 *v13; // rsi
  unsigned int v14; // ecx
  int v15; // edx
  char v16; // si
  char *v17; // rax
  unsigned int v18; // r8d
  int v19; // r13d
  _BYTE *v20; // rax
  int v21; // edx
  _BYTE *v22; // rax
  int v23; // [rsp+Ch] [rbp-E4h]
  int v25; // [rsp+10h] [rbp-E0h]
  unsigned int v26; // [rsp+10h] [rbp-E0h]
  unsigned int v27; // [rsp+10h] [rbp-E0h]
  unsigned int v28; // [rsp+10h] [rbp-E0h]
  int v29; // [rsp+10h] [rbp-E0h]
  _QWORD v31[8]; // [rsp+20h] [rbp-D0h] BYREF
  unsigned __int8 *v32; // [rsp+60h] [rbp-90h] BYREF
  size_t v33; // [rsp+68h] [rbp-88h]
  __int64 v34; // [rsp+70h] [rbp-80h]
  _BYTE v35[120]; // [rsp+78h] [rbp-78h] BYREF

  if ( !*((_DWORD *)a1 + 3) )
    return (void *)(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*a1 + 24LL))(*a1);
  v31[5] = 0x100000000LL;
  v31[0] = &unk_49DD288;
  v31[6] = &v32;
  v32 = v35;
  v33 = 0;
  v34 = 64;
  v31[1] = 2;
  memset(&v31[2], 0, 24);
  sub_CB5980((__int64)v31, 0, 0, 0);
  (*(void (__fastcall **)(_QWORD, _QWORD *, __int64, __int64))(*(_QWORD *)*a1 + 24LL))(*a1, v31, a3, a4);
  v6 = v33;
  v7 = *((unsigned int *)a1 + 3);
  if ( v7 > v33 )
  {
    v8 = *((_DWORD *)a1 + 2);
    v9 = v7 - v33;
    if ( !v8 )
    {
      v13 = v32;
      sub_CB6200(a2, v32, v33);
      v21 = 0;
      if ( v9 )
      {
        do
        {
          v13 = (unsigned __int8 *)*((unsigned __int8 *)a1 + 16);
          v22 = *(_BYTE **)(a2 + 32);
          if ( (unsigned __int64)v22 < *(_QWORD *)(a2 + 24) )
          {
            *(_QWORD *)(a2 + 32) = v22 + 1;
            *v22 = (_BYTE)v13;
          }
          else
          {
            v29 = v21;
            sub_CB5D20(a2, (char)v13);
            v21 = v29;
          }
          ++v21;
        }
        while ( v9 != v21 );
      }
      goto LABEL_13;
    }
    if ( v8 == 1 )
    {
      v14 = v9 >> 1;
      if ( v9 >> 1 )
      {
        v15 = 0;
        do
        {
          while ( 1 )
          {
            v16 = *((_BYTE *)a1 + 16);
            v17 = *(char **)(a2 + 32);
            if ( (unsigned __int64)v17 >= *(_QWORD *)(a2 + 24) )
              break;
            ++v15;
            *(_QWORD *)(a2 + 32) = v17 + 1;
            *v17 = v16;
            if ( v14 == v15 )
              goto LABEL_21;
          }
          v23 = v15;
          v26 = v14;
          sub_CB5D20(a2, v16);
          v14 = v26;
          v15 = v23 + 1;
        }
        while ( v26 != v23 + 1 );
LABEL_21:
        v6 = v33;
      }
      v13 = v32;
      v27 = v14;
      sub_CB6200(a2, v32, v6);
      v18 = v9 - v27;
      if ( v9 != v27 )
      {
        v19 = 0;
        do
        {
          v13 = (unsigned __int8 *)*((unsigned __int8 *)a1 + 16);
          v20 = *(_BYTE **)(a2 + 32);
          if ( (unsigned __int64)v20 < *(_QWORD *)(a2 + 24) )
          {
            *(_QWORD *)(a2 + 32) = v20 + 1;
            *v20 = (_BYTE)v13;
          }
          else
          {
            v28 = v18;
            sub_CB5D20(a2, (char)v13);
            v18 = v28;
          }
          ++v19;
        }
        while ( v18 != v19 );
      }
      goto LABEL_13;
    }
    if ( v9 )
    {
      v10 = 0;
      do
      {
        while ( 1 )
        {
          v11 = *((_BYTE *)a1 + 16);
          v12 = *(char **)(a2 + 32);
          if ( (unsigned __int64)v12 >= *(_QWORD *)(a2 + 24) )
            break;
          ++v10;
          *(_QWORD *)(a2 + 32) = v12 + 1;
          *v12 = v11;
          if ( v9 == v10 )
            goto LABEL_11;
        }
        v25 = v10;
        sub_CB5D20(a2, v11);
        v10 = v25 + 1;
      }
      while ( v9 != v25 + 1 );
LABEL_11:
      v6 = v33;
    }
  }
  v13 = v32;
  sub_CB6200(a2, v32, v6);
LABEL_13:
  v31[0] = &unk_49DD388;
  result = sub_CB5840((__int64)v31);
  if ( v32 != v35 )
    return (void *)_libc_free(v32, v13);
  return result;
}
