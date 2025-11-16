// Function: sub_2B09200
// Address: 0x2b09200
//
bool __fastcall sub_2B09200(_QWORD **a1, unsigned int a2)
{
  _DWORD *v2; // rax
  _DWORD *v3; // rdi
  unsigned __int64 v4; // rdx

  v2 = (_DWORD *)**a1;
  v3 = &v2[(*a1)[1]];
  if ( v2 != v3 )
  {
    v4 = 0;
    do
    {
      if ( *v2 != -1 && (a2 <= v4 || *v2 != (_DWORD)v4) )
        break;
      ++v2;
      ++v4;
    }
    while ( v3 != v2 );
  }
  return v3 == v2;
}
