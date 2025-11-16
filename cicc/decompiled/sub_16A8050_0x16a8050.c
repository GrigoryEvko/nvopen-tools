// Function: sub_16A8050
// Address: 0x16a8050
//
void __fastcall sub_16A8050(_QWORD *a1, unsigned int a2, unsigned int a3)
{
  _QWORD *v3; // r8
  unsigned int v4; // eax
  __int64 v5; // rbx
  size_t v6; // r12
  char v7; // r9
  __int64 v8; // r10
  __int64 v9; // rdx

  if ( a3 )
  {
    v3 = a1;
    v4 = a3 >> 6;
    if ( a3 >> 6 > a2 )
      v4 = a2;
    v5 = a2 - v4;
    v6 = v4;
    v7 = a3 & 0x3F;
    if ( (a3 & 0x3F) != 0 )
    {
      if ( (_DWORD)v5 )
      {
        v8 = v4;
        while ( 1 )
        {
          ++v4;
          v9 = v3[v8] >> v7;
          *a1 = v9;
          if ( a2 == v4 )
            break;
          v8 = v4;
          *a1++ = (v3[v4] << (64 - v7)) | v9;
        }
      }
    }
    else
    {
      v3 = memmove(a1, &a1[v6], (unsigned int)(8 * v5));
    }
    memset(&v3[v5], 0, v6 * 8);
  }
}
