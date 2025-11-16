// Function: sub_30F50C0
// Address: 0x30f50c0
//
__int16 __fastcall sub_30F50C0(__int64 a1, __int64 a2, unsigned int a3, _QWORD **a4, __int64 a5, __int64 a6)
{
  __int16 result; // ax
  _QWORD *v10; // rax
  int v11; // edx
  int v12; // ebx
  int v13; // r15d
  int v14; // r12d
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned int v17; // r14d
  char v18; // bl
  void (*v19)(void); // rax
  __int64 *v20; // rax
  __int64 v21; // rax
  char v22; // [rsp+Fh] [rbp-41h]
  unsigned __int64 v23[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( *(_QWORD *)(a1 + 16) != *(_QWORD *)(a2 + 16) && !sub_30F4F20(a1, a2, a6) )
    return 256;
  sub_2297CA0((__int64 *)v23, a5, *(_QWORD *)(a1 + 8), *(_BYTE **)(a2 + 8));
  if ( v23[0] )
  {
    if ( !(*(unsigned __int8 (__fastcall **)(unsigned __int64))(*(_QWORD *)v23[0] + 16LL))(v23[0]) )
    {
      v10 = *a4;
      if ( *a4 )
      {
        v11 = 1;
        do
        {
          v10 = (_QWORD *)*v10;
          ++v11;
        }
        while ( v10 );
        v12 = v11;
      }
      else
      {
        v12 = 1;
      }
      v13 = 1;
      v14 = (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v23[0] + 40LL))(v23[0]);
      if ( v14 > 0 )
      {
        do
        {
          v15 = (*(__int64 (__fastcall **)(unsigned __int64, _QWORD))(*(_QWORD *)v23[0] + 56LL))(
                  v23[0],
                  (unsigned int)v13);
          if ( !v15 || *(_WORD *)(v15 + 24) )
          {
            v18 = 0;
            goto LABEL_18;
          }
          v16 = *(_QWORD *)(v15 + 32);
          v17 = *(_DWORD *)(v16 + 32);
          if ( v12 == v13 )
          {
            v20 = *(__int64 **)(v16 + 24);
            if ( v17 <= 0x40 )
            {
              if ( !v17 )
                goto LABEL_16;
              v21 = (__int64)((_QWORD)v20 << (64 - (unsigned __int8)v17)) >> (64 - (unsigned __int8)v17);
            }
            else
            {
              v21 = *v20;
            }
            if ( a3 < v21 )
            {
LABEL_30:
              v22 = 0;
              v18 = 1;
              goto LABEL_18;
            }
          }
          else if ( v17 <= 0x40 )
          {
            if ( *(_QWORD *)(v16 + 24) )
              goto LABEL_30;
          }
          else if ( (unsigned int)sub_C444A0(v16 + 24) != v17 )
          {
            goto LABEL_30;
          }
LABEL_16:
          ++v13;
        }
        while ( v14 >= v13 );
      }
    }
    v22 = 1;
    v18 = 1;
LABEL_18:
    if ( v23[0] )
    {
      v19 = *(void (**)(void))(*(_QWORD *)v23[0] + 8LL);
      if ( (char *)v19 == (char *)sub_228A6E0 )
        j_j___libc_free_0(v23[0]);
      else
        v19();
    }
  }
  else
  {
    v22 = 0;
    v18 = 1;
  }
  LOBYTE(result) = v22;
  HIBYTE(result) = v18;
  return result;
}
