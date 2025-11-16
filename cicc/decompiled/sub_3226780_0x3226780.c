// Function: sub_3226780
// Address: 0x3226780
//
void __fastcall sub_3226780(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // rdx
  __int64 v5; // rcx
  _QWORD *v6; // r13
  __int64 v7; // rbx
  _QWORD *v8; // r12

  v2 = *(_QWORD **)a1;
  v3 = 2LL * *(unsigned int *)(a1 + 8);
  if ( v3 * 8 )
  {
    v4 = &a2[v3];
    do
    {
      if ( a2 )
      {
        *a2 = *v2;
        v5 = v2[1];
        *v2 = 0;
        a2[1] = v5;
      }
      a2 += 2;
      v2 += 2;
    }
    while ( v4 != a2 );
    v6 = *(_QWORD **)a1;
    v7 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    while ( v6 != (_QWORD *)v7 )
    {
      while ( 1 )
      {
        v8 = *(_QWORD **)(v7 - 16);
        v7 -= 16;
        if ( !v8 )
          break;
        *v8 = &unk_4A35D40;
        sub_32478E0(v8);
        j_j___libc_free_0((unsigned __int64)v8);
        if ( v6 == (_QWORD *)v7 )
          return;
      }
    }
  }
}
