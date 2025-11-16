// Function: sub_2E33470
// Address: 0x2e33470
//
void __fastcall sub_2E33470(unsigned int *a1, unsigned int *a2)
{
  unsigned int *v2; // rcx
  unsigned int v3; // edx
  unsigned __int64 v4; // r8
  unsigned int *v5; // r12
  unsigned int *v7; // rsi
  unsigned int *v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  unsigned int *v12; // rdi
  unsigned int v13; // eax
  unsigned int v14; // [rsp-1Ch] [rbp-1Ch] BYREF

  if ( a2 == a1 )
    return;
  v2 = a1;
  v3 = 0;
  v4 = 0;
  v5 = a1;
  v7 = a1;
  while ( 1 )
  {
    v9 = *v2;
    if ( (_DWORD)v9 == -1 )
      break;
    v4 += v9;
    v8 = v2 + 1;
    if ( a2 == v2 + 1 )
      goto LABEL_7;
LABEL_4:
    v2 = v8;
  }
  v8 = v2 + 1;
  ++v3;
  if ( a2 != v2 + 1 )
    goto LABEL_4;
LABEL_7:
  if ( !v3 )
  {
    if ( !v4 )
    {
      sub_F02DB0(&v14, 1u, a2 - a1);
      v13 = v14;
      do
        *v5++ = v13;
      while ( a2 != v5 );
      return;
    }
    do
    {
LABEL_17:
      v11 = *v7;
      v12 = v7++;
      *(v7 - 1) = ((v4 >> 1) + (v11 << 31)) / v4;
    }
    while ( v2 != v12 );
    return;
  }
  LODWORD(v10) = 0;
  if ( v4 < 0x80000000 )
    v10 = (0x80000000 - v4) / v3;
  while ( 1 )
  {
    if ( *a1 == -1 )
      *a1 = v10;
    if ( a1 == v2 )
      break;
    ++a1;
  }
  if ( v4 > 0x80000000 )
    goto LABEL_17;
}
