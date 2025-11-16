// Function: sub_1ED7BB0
// Address: 0x1ed7bb0
//
__int64 __fastcall sub_1ED7BB0(__int64 a1, unsigned int a2)
{
  _QWORD *v3; // rdi
  __int64 ***v4; // r14
  __int64 **v5; // r13
  __int64 **v6; // rbx
  int *v7; // rax
  int v8; // edx
  unsigned int v9; // eax
  int v10; // r14d
  _DWORD *v11; // rbx
  int v12; // r14d
  int v13; // ebx
  unsigned int v15; // [rsp+4h] [rbp-3Ch]
  __int64 ***v16; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD **)(a1 + 24);
  v16 = (__int64 ***)v3[33];
  if ( v16 == (__int64 ***)v3[32] )
  {
    sub_1ED7890(a1, 0);
    BUG();
  }
  v15 = 0;
  v4 = (__int64 ***)v3[32];
  v5 = 0;
  while ( 1 )
  {
    v6 = *v4;
    v7 = (int *)(*(__int64 (__fastcall **)(_QWORD *, __int64 **))(*v3 + 224LL))(v3, *v4);
    v8 = *v7;
    if ( *v7 != -1 )
    {
      while ( v8 != a2 )
      {
        v8 = v7[1];
        ++v7;
        if ( v8 == -1 )
          goto LABEL_9;
      }
      v9 = *(_DWORD *)((*(__int64 (__fastcall **)(_QWORD, __int64 **))(**(_QWORD **)(a1 + 24) + 184LL))(
                         *(_QWORD *)(a1 + 24),
                         v6)
                     + 4);
      if ( !v5 || v9 > v15 )
        break;
    }
LABEL_9:
    if ( v16 == ++v4 )
      goto LABEL_12;
LABEL_10:
    v3 = *(_QWORD **)(a1 + 24);
  }
  v15 = v9;
  v5 = v6;
  if ( v16 != ++v4 )
    goto LABEL_10;
LABEL_12:
  sub_1ED7890(a1, v5);
  v10 = *((unsigned __int16 *)*v5 + 10);
  v11 = (_DWORD *)(*(_QWORD *)a1 + 24LL * *((unsigned __int16 *)*v5 + 12));
  if ( *(_DWORD *)(a1 + 8) != *v11 )
    sub_1ED7890(a1, v5);
  v12 = v10 - v11[1];
  v13 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 24) + 216LL))(
          *(_QWORD *)(a1 + 24),
          *(_QWORD *)(a1 + 16),
          a2);
  return (unsigned int)(v13
                      - *(_DWORD *)(*(__int64 (__fastcall **)(_QWORD, __int64 **))(**(_QWORD **)(a1 + 24) + 184LL))(
                                     *(_QWORD *)(a1 + 24),
                                     v5)
                      * v12);
}
