// Function: sub_32645C0
// Address: 0x32645c0
//
__int64 __fastcall sub_32645C0(unsigned int *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r14
  __int64 v5; // r12
  unsigned int v6; // ebx
  unsigned __int64 v7; // rbx
  unsigned int v8; // r12d
  __int64 v9; // r15
  unsigned int v10; // r13d
  unsigned __int64 v11; // rax
  unsigned __int64 *v13; // rax
  unsigned __int64 v14; // rax

  v4 = *a1;
  v5 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v6 = *(_DWORD *)(v5 + 32);
  if ( v6 <= 0x40 )
  {
    v7 = *(_QWORD *)(v5 + 24);
    v8 = 0;
    if ( v4 <= v7 )
      return v8;
    goto LABEL_3;
  }
  if ( v6 - (unsigned int)sub_C444A0(v5 + 24) > 0x40 )
    return 0;
  v13 = *(unsigned __int64 **)(v5 + 24);
  v8 = 0;
  v7 = *v13;
  if ( v4 > *v13 )
  {
LABEL_3:
    v9 = *(_QWORD *)(*(_QWORD *)a3 + 96LL);
    v10 = *(_DWORD *)(v9 + 32);
    if ( v10 > 0x40 )
    {
      if ( v10 - (unsigned int)sub_C444A0(v9 + 24) <= 0x40 )
      {
        v8 = 0;
        v14 = **(_QWORD **)(v9 + 24);
        if ( v4 > v14 )
          LOBYTE(v8) = v7 <= v14;
      }
    }
    else
    {
      v11 = *(_QWORD *)(v9 + 24);
      v8 = 0;
      if ( v4 > v11 )
        LOBYTE(v8) = v7 <= v11;
    }
  }
  return v8;
}
