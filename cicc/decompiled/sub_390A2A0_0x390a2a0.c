// Function: sub_390A2A0
// Address: 0x390a2a0
//
unsigned __int64 __fastcall sub_390A2A0(__int64 a1)
{
  __int64 v1; // rdi
  __int64 (*v2)(void); // rax
  __int64 v3; // r13
  unsigned __int64 result; // rax
  _QWORD **v5; // rax
  __int64 v6; // rdx
  _QWORD **v7; // r14
  _QWORD *v8; // rbx
  _QWORD **v9; // r12
  __int64 (__fastcall *v10)(); // rax
  _QWORD **v11; // rax
  char v12; // [rsp-39h] [rbp-39h]

  v1 = *(_QWORD *)(a1 + 8);
  if ( !*(_QWORD *)(v1 + 160) )
    return nullsub_2029();
  v12 = 0;
  v2 = *(__int64 (**)(void))(*(_QWORD *)v1 + 8LL);
  if ( v2 != sub_390D9C0 )
    v12 = v2();
  v3 = 0;
  if ( *(_BYTE *)(v1 + 8) )
  {
    v5 = *(_QWORD ***)(v1 + 32);
    if ( v5 == *(_QWORD ***)(v1 + 24) )
      v6 = *(unsigned int *)(v1 + 44);
    else
      v6 = *(unsigned int *)(v1 + 40);
    v7 = &v5[v6];
    if ( v5 == v7 )
    {
LABEL_17:
      v3 = 0;
    }
    else
    {
      while ( 1 )
      {
        v8 = *v5;
        v9 = v5;
        if ( (unsigned __int64)*v5 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v7 == ++v5 )
          goto LABEL_17;
      }
      v3 = 0;
      if ( v5 != v7 )
      {
        v10 = *(__int64 (__fastcall **)())(*v8 + 32LL);
        if ( v10 != sub_390D9F0 )
          goto LABEL_27;
        while ( 1 )
        {
          v11 = v9 + 1;
          if ( v9 + 1 == v7 )
            break;
          while ( 1 )
          {
            v8 = *v11;
            v9 = v11;
            if ( (unsigned __int64)*v11 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v7 == ++v11 )
              goto LABEL_5;
          }
          if ( v11 == v7 )
            break;
          v10 = *(__int64 (__fastcall **)())(*v8 + 32LL);
          if ( v10 != sub_390D9F0 )
          {
LABEL_27:
            if ( ((unsigned __int8 (__fastcall *)(_QWORD *))v10)(v8) )
              v3 |= v8[1];
          }
        }
      }
    }
  }
LABEL_5:
  result = sub_38D4B30(*(_QWORD *)(v1 + 160));
  if ( result && *(_BYTE *)(result + 16) == 9 || v3 || v12 )
  {
    result = sub_38D57A0(*(_QWORD *)(v1 + 160));
    *(_QWORD *)(v1 + 88) = result;
    if ( v12 )
    {
      *(_BYTE *)(result + 56) = 1;
      result = *(_QWORD *)(v1 + 88);
    }
    *(_QWORD *)(result + 48) |= v3;
  }
  return result;
}
