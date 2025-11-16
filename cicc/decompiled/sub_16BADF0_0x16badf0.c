// Function: sub_16BADF0
// Address: 0x16badf0
//
void __fastcall sub_16BADF0(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // rax

  if ( a1 )
  {
    v2 = *(_QWORD **)(a1 + 88);
    v3 = *(_QWORD **)(a1 + 80);
    if ( v2 != v3 )
    {
      do
      {
        if ( (_QWORD *)*v3 != v3 + 2 )
          j_j___libc_free_0(*v3, v3[2] + 1LL);
        v3 += 4;
      }
      while ( v2 != v3 );
      v3 = *(_QWORD **)(a1 + 80);
    }
    if ( v3 )
      j_j___libc_free_0(v3, *(_QWORD *)(a1 + 96) - (_QWORD)v3);
    v4 = *(_QWORD *)(a1 + 48);
    while ( v4 )
    {
      v5 = v4;
      sub_16BAAF0(*(_QWORD **)(v4 + 24));
      v6 = *(_QWORD *)(v4 + 32);
      v4 = *(_QWORD *)(v4 + 16);
      if ( v6 != v5 + 48 )
        j_j___libc_free_0(v6, *(_QWORD *)(v5 + 48) + 1LL);
      j_j___libc_free_0(v5, 72);
    }
    v7 = *(unsigned int *)(a1 + 24);
    if ( (_DWORD)v7 )
    {
      v8 = *(_QWORD *)(a1 + 8);
      v9 = v8 + 72 * v7;
      do
      {
        while ( 1 )
        {
          if ( *(_DWORD *)v8 <= 0xFFFFFFFD )
          {
            v10 = *(_QWORD *)(v8 + 40);
            if ( v10 != v8 + 56 )
              break;
          }
          v8 += 72;
          if ( v9 == v8 )
            goto LABEL_19;
        }
        v11 = *(_QWORD *)(v8 + 56);
        v8 += 72;
        j_j___libc_free_0(v10, v11 + 1);
      }
      while ( v9 != v8 );
    }
LABEL_19:
    j___libc_free_0(*(_QWORD *)(a1 + 8));
    j_j___libc_free_0(a1, 112);
  }
}
