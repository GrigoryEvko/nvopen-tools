// Function: sub_228D790
// Address: 0x228d790
//
void __fastcall sub_228D790(__int64 a1, __int64 a2, _QWORD **a3, unsigned __int64 *a4)
{
  _QWORD *v4; // r14
  _QWORD *v6; // rdx
  unsigned int v7; // ecx
  _QWORD *v8; // rax
  unsigned int v9; // ebx
  unsigned __int64 v10; // rax

  if ( a3 )
  {
    v4 = a3;
    v6 = *a3;
    v7 = *(_DWORD *)(a1 + 32);
    if ( v6 )
      goto LABEL_3;
    while ( v7 )
    {
      v9 = 1;
LABEL_10:
      if ( !sub_DADE90(*(_QWORD *)(a1 + 8), a2, (__int64)v4) )
      {
        v10 = *a4;
        if ( (*a4 & 1) != 0 )
          *a4 = 2 * ((v10 >> 58 << 57) | ~(-1LL << (v10 >> 58)) & (~(-1LL << (v10 >> 58)) & (v10 >> 1) | (1LL << v9)))
              + 1;
        else
          *(_QWORD *)(*(_QWORD *)v10 + 8LL * (v9 >> 6)) |= 1LL << v9;
      }
      v6 = (_QWORD *)*v4;
      if ( !*v4 )
        break;
      v7 = *(_DWORD *)(a1 + 32);
      while ( 1 )
      {
        v4 = v6;
        v6 = (_QWORD *)*v6;
        if ( !v6 )
          break;
LABEL_3:
        v8 = v6;
        v9 = 1;
        do
        {
          v8 = (_QWORD *)*v8;
          ++v9;
        }
        while ( v8 );
        if ( v9 <= v7 )
          goto LABEL_10;
      }
    }
  }
}
