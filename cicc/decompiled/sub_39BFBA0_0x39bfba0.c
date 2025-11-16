// Function: sub_39BFBA0
// Address: 0x39bfba0
//
void __fastcall sub_39BFBA0(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // r9d
  unsigned __int64 v6; // r12
  unsigned int **v7; // rdi
  unsigned int **v8; // rbx
  unsigned int **v9; // r12
  unsigned int *v10; // r13
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 *v14; // r12
  __int64 *v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rcx
  unsigned int **v19; // [rsp+10h] [rbp-240h] BYREF
  __int64 v20; // [rsp+18h] [rbp-238h]
  _BYTE s[560]; // [rsp+20h] [rbp-230h] BYREF

  if ( (unsigned __int16)sub_3971A70(a2) > 4u )
    sub_39BFAD0(a1, a2, a3);
  if ( *(_DWORD *)(a1 + 16) )
  {
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 256) + 160LL))(*(_QWORD *)(a2 + 256), a3, 0);
    v6 = *(unsigned int *)(a1 + 16);
    v7 = (unsigned int **)s;
    v19 = (unsigned int **)s;
    v20 = 0x4000000000LL;
    if ( (unsigned int)v6 > 0x40 )
    {
      sub_16CD150((__int64)&v19, s, v6, 8, (int)&v19, v5);
      v7 = v19;
    }
    LODWORD(v20) = v6;
    if ( 8 * v6 )
      memset(v7, 0, 8 * v6);
    if ( *(_DWORD *)(a1 + 16) )
    {
      v13 = *(__int64 **)(a1 + 8);
      v14 = &v13[2 * *(unsigned int *)(a1 + 24)];
      if ( v13 != v14 )
      {
        while ( 1 )
        {
          v15 = v13;
          if ( *v13 != -8 && *v13 != -16 )
            break;
          v13 += 2;
          if ( v14 == v13 )
            goto LABEL_9;
        }
        while ( v14 != v15 )
        {
          if ( *((_BYTE *)v15 + 12) )
          {
            v16 = sub_396DD80(a2);
            v17 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v16 + 112LL))(v16, *v15);
          }
          else
          {
            v17 = sub_38CF310(*v15, 0, *(_QWORD *)(a2 + 248), 0);
          }
          v18 = *((unsigned int *)v15 + 2);
          v15 += 2;
          v19[v18] = (unsigned int *)v17;
          if ( v15 == v14 )
            break;
          while ( *v15 == -16 || *v15 == -8 )
          {
            v15 += 2;
            if ( v14 == v15 )
              goto LABEL_9;
          }
        }
      }
    }
LABEL_9:
    v8 = v19;
    v9 = &v19[(unsigned int)v20];
    if ( v19 != v9 )
    {
      do
      {
        v10 = *v8;
        v11 = *(_QWORD *)(a2 + 256);
        ++v8;
        v12 = sub_396DDB0(a2);
        sub_15A9520(v12, 0);
        sub_38DDD30(v11, v10);
      }
      while ( v9 != v8 );
      v9 = v19;
    }
    if ( v9 != (unsigned int **)s )
      _libc_free((unsigned __int64)v9);
  }
}
