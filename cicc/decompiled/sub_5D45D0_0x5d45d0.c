// Function: sub_5D45D0
// Address: 0x5d45d0
//
int __fastcall sub_5D45D0(unsigned int *a1)
{
  unsigned int v1; // r12d
  int result; // eax
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rsi
  int v6; // ebx
  FILE *v7; // r8
  unsigned int v8; // [rsp+8h] [rbp-18h] BYREF
  _DWORD v9[5]; // [rsp+Ch] [rbp-14h] BYREF

  v1 = *a1;
  result = dword_4CF7EA0;
  if ( *a1 )
  {
    unk_4F07508 = *(_QWORD *)a1;
    if ( dword_4CF7EA0 && dword_4CF7F40 )
      ((void (*)(void))sub_5D37C0)();
    v3 = sub_729B10(v1, &v8, v9, 0);
    v4 = v8;
    v5 = v3;
    if ( v9[0] && !v8 )
    {
      v8 = 1;
      v4 = 1;
    }
    if ( qword_4CF7F48 != v3
      || !dword_4CF7F3C
      || (result = dword_4CF7F44, dword_4CF7F44 > (unsigned int)v4)
      || dword_4CF7F44 + 5 < (unsigned int)v4
      || (v7 = stream, qword_4CF7EB8 == stream) )
    {
      result = sub_5D43B0(v4, v5);
      if ( !dword_4CF7EA0 )
        return result;
      goto LABEL_15;
    }
    if ( dword_4CF7F44 != (_DWORD)v4 )
    {
      do
      {
        sub_5D37C0(v4, v5);
        result = dword_4CF7F44;
      }
      while ( v8 > dword_4CF7F44 );
      if ( !dword_4CF7EA0 )
        return result;
      goto LABEL_15;
    }
    if ( !dword_4CF7F40 )
    {
      if ( !dword_4CF7EA0 )
        return result;
      goto LABEL_15;
    }
LABEL_28:
    result = putc(32, v7);
    ++dword_4CF7F40;
    return result;
  }
  if ( !dword_4CF7F3C )
  {
    if ( !dword_4CF7EA0 )
      return result;
    goto LABEL_15;
  }
  if ( !dword_4CF7EA0 )
  {
    result = dword_4CF7F40;
    v7 = stream;
    if ( !dword_4CF7F40 )
      return result;
    goto LABEL_28;
  }
LABEL_15:
  result = dword_4CF7F38;
  if ( dword_4CF7F38 > 0 )
  {
    v6 = 0;
    do
    {
      ++v6;
      putc(32, stream);
      result = dword_4CF7F38;
    }
    while ( dword_4CF7F38 > v6 );
  }
  dword_4CF7F40 += result;
  return result;
}
