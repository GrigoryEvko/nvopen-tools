// Function: sub_16E8170
// Address: 0x16e8170
//
void __fastcall sub_16E8170(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // rbx
  size_t v6; // rdx
  int v7; // eax
  unsigned __int64 v8; // rbx
  int v9; // ecx
  int v10; // ebx
  char v11; // si
  char *v12; // rax
  int v13; // ecx
  int v14; // r15d
  char v15; // si
  char *v16; // rax
  int v17; // ebx
  int i; // r15d
  char v19; // si
  char *v20; // rax
  int v21; // r15d
  int v22; // ecx
  char v23; // si
  char *v24; // rax
  int v26; // [rsp+0h] [rbp-D0h]
  int v27; // [rsp+0h] [rbp-D0h]
  int v28; // [rsp+0h] [rbp-D0h]
  _QWORD v30[4]; // [rsp+20h] [rbp-B0h] BYREF
  int v31; // [rsp+40h] [rbp-90h]
  char **v32; // [rsp+48h] [rbp-88h]
  char *v33; // [rsp+50h] [rbp-80h] BYREF
  __int64 v34; // [rsp+58h] [rbp-78h]
  _BYTE v35[112]; // [rsp+60h] [rbp-70h] BYREF

  if ( !a1[2] )
  {
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*a1 + 16LL))(*a1);
    return;
  }
  v34 = 0x4000000000LL;
  v30[0] = &unk_49EFC48;
  v33 = v35;
  v32 = &v33;
  v31 = 1;
  memset(&v30[1], 0, 24);
  sub_16E7A40((__int64)v30, 0, 0, 0);
  (*(void (__fastcall **)(_QWORD, _QWORD *, __int64, __int64))(*(_QWORD *)*a1 + 16LL))(*a1, v30, a3, a4);
  v5 = a1[2];
  v6 = (unsigned int)v34;
  if ( v5 > (unsigned int)v34 )
  {
    v7 = *((_DWORD *)a1 + 2);
    v8 = v5 - (unsigned int)v34;
    v9 = v8;
    if ( !v7 )
    {
      v21 = 0;
      sub_16E7EE0(a2, v33, (unsigned int)v34);
      v22 = v8;
      if ( (_DWORD)v8 )
      {
        do
        {
          v23 = *((_BYTE *)a1 + 24);
          v24 = *(char **)(a2 + 24);
          if ( (unsigned __int64)v24 < *(_QWORD *)(a2 + 16) )
          {
            *(_QWORD *)(a2 + 24) = v24 + 1;
            *v24 = v23;
          }
          else
          {
            v28 = v22;
            sub_16E7DE0(a2, v23);
            v22 = v28;
          }
          ++v21;
        }
        while ( v21 != v22 );
      }
      goto LABEL_13;
    }
    if ( v7 == 1 )
    {
      v13 = v8 >> 1;
      if ( v13 )
      {
        v14 = 0;
        do
        {
          while ( 1 )
          {
            v15 = *((_BYTE *)a1 + 24);
            v16 = *(char **)(a2 + 24);
            if ( (unsigned __int64)v16 >= *(_QWORD *)(a2 + 16) )
              break;
            ++v14;
            *(_QWORD *)(a2 + 24) = v16 + 1;
            *v16 = v15;
            if ( v13 == v14 )
              goto LABEL_21;
          }
          v27 = v13;
          ++v14;
          sub_16E7DE0(a2, v15);
          v13 = v27;
        }
        while ( v27 != v14 );
LABEL_21:
        v6 = (unsigned int)v34;
      }
      sub_16E7EE0(a2, v33, v6);
      v17 = v8 - (v8 >> 1);
      if ( v17 )
      {
        for ( i = 0; i != v17; ++i )
        {
          v19 = *((_BYTE *)a1 + 24);
          v20 = *(char **)(a2 + 24);
          if ( (unsigned __int64)v20 < *(_QWORD *)(a2 + 16) )
          {
            *(_QWORD *)(a2 + 24) = v20 + 1;
            *v20 = v19;
          }
          else
          {
            sub_16E7DE0(a2, v19);
          }
        }
      }
      goto LABEL_13;
    }
    if ( (_DWORD)v8 )
    {
      v10 = 0;
      do
      {
        while ( 1 )
        {
          v11 = *((_BYTE *)a1 + 24);
          v12 = *(char **)(a2 + 24);
          if ( (unsigned __int64)v12 >= *(_QWORD *)(a2 + 16) )
            break;
          ++v10;
          *(_QWORD *)(a2 + 24) = v12 + 1;
          *v12 = v11;
          if ( v10 == v9 )
            goto LABEL_11;
        }
        v26 = v9;
        ++v10;
        sub_16E7DE0(a2, v11);
        v9 = v26;
      }
      while ( v10 != v26 );
LABEL_11:
      v6 = (unsigned int)v34;
    }
  }
  sub_16E7EE0(a2, v33, v6);
LABEL_13:
  v30[0] = &unk_49EFD28;
  sub_16E7960((__int64)v30);
  if ( v33 != v35 )
    _libc_free((unsigned __int64)v33);
}
