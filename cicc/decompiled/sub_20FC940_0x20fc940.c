// Function: sub_20FC940
// Address: 0x20fc940
//
void __fastcall sub_20FC940(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  unsigned __int64 v6; // r8
  unsigned int v8; // ebx
  unsigned __int64 v9; // rdi

  v6 = *(_QWORD *)(a1 + 8);
  if ( v6 )
  {
    if ( *(_DWORD *)a1 )
    {
      v8 = 0;
      do
      {
        while ( 1 )
        {
          v9 = v6 + 216LL * v8;
          if ( *(_DWORD *)(v9 + 200) )
            break;
          if ( *(_DWORD *)a1 == ++v8 )
            goto LABEL_7;
        }
        ++v8;
        sub_20FC410(v9 + 8, (char *)sub_20FC1C0, 0, a4, v6, a6);
        v6 = *(_QWORD *)(a1 + 8);
      }
      while ( *(_DWORD *)a1 != v8 );
    }
LABEL_7:
    _libc_free(v6);
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
  }
}
