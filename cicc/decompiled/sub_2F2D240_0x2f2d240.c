// Function: sub_2F2D240
// Address: 0x2f2d240
//
void __fastcall sub_2F2D240(_QWORD *a1, unsigned int *a2, unsigned int *a3)
{
  _QWORD *v3; // r14
  unsigned int *v4; // rbx
  char v5; // r15
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // r12

  if ( a3 != a2 )
  {
    v3 = a1 + 1;
    v4 = a2;
    do
    {
      v7 = sub_2DCC990(a1, (__int64)v3, v4);
      v9 = (_QWORD *)v8;
      if ( v8 )
      {
        v5 = v7 || v3 == (_QWORD *)v8 || *v4 < *(_DWORD *)(v8 + 32);
        v6 = sub_22077B0(0x28u);
        *(_DWORD *)(v6 + 32) = *v4;
        sub_220F040(v5, v6, v9, v3);
        ++a1[5];
      }
      ++v4;
    }
    while ( a3 != v4 );
  }
}
