// Function: sub_27DE390
// Address: 0x27de390
//
void __fastcall sub_27DE390(unsigned int *a1, unsigned int *a2)
{
  unsigned int v2; // edx
  unsigned __int64 v3; // rcx
  unsigned int *v4; // r12
  unsigned int *v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned int *v8; // rdx
  __int64 v9; // rax
  unsigned int v10; // eax
  unsigned int v11; // [rsp-1Ch] [rbp-1Ch] BYREF

  if ( a1 == a2 )
    return;
  v2 = 0;
  v3 = 0;
  v4 = a1;
  v5 = a1;
  do
  {
    while ( 1 )
    {
      v6 = *v5;
      if ( (_DWORD)v6 == -1 )
        break;
      ++v5;
      v3 += v6;
      if ( a2 == v5 )
        goto LABEL_6;
    }
    ++v5;
    ++v2;
  }
  while ( a2 != v5 );
LABEL_6:
  if ( !v2 )
  {
    if ( !v3 )
    {
      sub_F02DB0(&v11, 1u, v5 - a1);
      v10 = v11;
      do
        *v4++ = v10;
      while ( v5 != v4 );
      return;
    }
    do
    {
LABEL_16:
      v9 = *v4++;
      *(v4 - 1) = ((v3 >> 1) + (v9 << 31)) / v3;
    }
    while ( v4 != v5 );
    return;
  }
  LODWORD(v7) = 0;
  if ( v3 < 0x80000000 )
    v7 = (0x80000000 - v3) / v2;
  v8 = a1;
  do
  {
    if ( *v8 == -1 )
      *v8 = v7;
    ++v8;
  }
  while ( v5 != v8 );
  if ( v3 > 0x80000000 )
    goto LABEL_16;
}
