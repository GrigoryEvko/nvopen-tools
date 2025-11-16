// Function: sub_5D3B20
// Address: 0x5d3b20
//
__int64 __fastcall sub_5D3B20(FILE *a1)
{
  __int64 v1; // rdx
  __int64 *v2; // rax
  __int64 *v3; // rax
  int v4; // edx
  __int64 result; // rax

  v1 = qword_4CF7F58;
  if ( stream )
  {
    if ( stream == (FILE *)qword_4CF7F58 )
    {
      v2 = (__int64 *)&unk_4CF7F20;
    }
    else if ( stream == qword_4CF7EB8 )
    {
      v2 = &qword_4CF7EE0;
    }
    else if ( stream == qword_4CF7EB0 )
    {
      v2 = (__int64 *)&unk_4CF7EC0;
    }
    else
    {
      if ( stream != qword_4CF7EA8 )
      {
        MEMORY[0] = qword_4CF7F48;
        BUG();
      }
      v2 = &qword_4CF7F00;
    }
    *v2 = qword_4CF7F48;
    *((_DWORD *)v2 + 2) = dword_4CF7F44;
    *((_DWORD *)v2 + 3) = dword_4CF7F40;
    *((_DWORD *)v2 + 4) = dword_4CF7F3C;
  }
  stream = a1;
  if ( a1 == (FILE *)v1 )
  {
    v3 = (__int64 *)&unk_4CF7F20;
  }
  else if ( a1 == qword_4CF7EB8 )
  {
    v3 = &qword_4CF7EE0;
  }
  else if ( a1 == qword_4CF7EB0 )
  {
    v3 = (__int64 *)&unk_4CF7EC0;
  }
  else
  {
    if ( a1 != qword_4CF7EA8 )
      BUG();
    v3 = &qword_4CF7F00;
  }
  qword_4CF7F48 = *v3;
  dword_4CF7F44 = *((_DWORD *)v3 + 2);
  v4 = *((_DWORD *)v3 + 3);
  result = *((unsigned int *)v3 + 4);
  dword_4CF7F40 = v4;
  dword_4CF7F3C = result;
  return result;
}
