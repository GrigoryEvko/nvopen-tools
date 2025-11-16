// Function: sub_2E19FF0
// Address: 0x2e19ff0
//
void __fastcall sub_2E19FF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  unsigned __int64 v7; // rdi
  unsigned int v8; // r13d
  unsigned __int64 v9; // r12
  _QWORD *v10; // rax

  v7 = *(_QWORD *)(a1 + 8);
  if ( v7 )
  {
    if ( *(_DWORD *)a1 )
    {
      v8 = 0;
      do
      {
        while ( 1 )
        {
          v9 = v7 + 216LL * v8;
          if ( *(_DWORD *)(v9 + 200) )
            break;
          if ( *(_DWORD *)a1 == ++v8 )
            goto LABEL_9;
        }
        sub_2E19AD0(v9 + 8, (char *)sub_2E199D0, 0, a4, a5, a6);
        v10 = (_QWORD *)(v9 + 8);
        do
        {
          *v10 = 0;
          v10 += 2;
          *(v10 - 1) = 0;
        }
        while ( v10 != (_QWORD *)(v9 + 136) );
        v7 = *(_QWORD *)(a1 + 8);
        ++v8;
      }
      while ( *(_DWORD *)a1 != v8 );
    }
LABEL_9:
    _libc_free(v7);
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
  }
}
