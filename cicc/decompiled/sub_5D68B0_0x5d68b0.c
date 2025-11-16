// Function: sub_5D68B0
// Address: 0x5d68b0
//
int __fastcall sub_5D68B0(__int64 a1, __int64 a2)
{
  FILE *v3; // r12
  FILE *v4; // rdi
  char *v5; // rbx
  int v6; // edi
  __int64 v7; // rdi
  char *v8; // rbx
  int v9; // edi
  int v10; // edi
  char *v11; // rbx
  __int64 v12; // rdi
  char *v13; // r13
  int v14; // edi
  int result; // eax
  FILE *v16; // rax

  v3 = stream;
  if ( !dword_4CF7CD4 )
  {
    v4 = qword_4CF7EA8;
    if ( !qword_4CF7EA8 )
    {
      v16 = (FILE *)sub_721330(0, a2);
      qword_4CF7F00 = 0;
      qword_4CF7EA8 = v16;
      v4 = v16;
      qword_4CF7F08 = 0;
      dword_4CF7F10 = 0;
    }
    sub_5D3B20(v4);
  }
  v5 = "emset((char *)";
  sub_5D45D0((unsigned int *)(a1 + 64));
  v6 = 109;
  do
  {
    ++v5;
    putc(v6, stream);
    v6 = *(v5 - 1);
  }
  while ( *(v5 - 1) );
  v7 = *(_QWORD *)(a1 + 120);
  dword_4CF7F40 += 15;
  v8 = " 0";
  sub_5D48C0(v7);
  sub_5D6390(a1);
  v9 = 44;
  do
  {
    ++v8;
    putc(v9, stream);
    v9 = *(v8 - 1);
  }
  while ( *(v8 - 1) );
  dword_4CF7F40 += 3;
  v10 = 44;
  v11 = "sizeof(";
  do
  {
    ++v11;
    putc(v10, stream);
    v10 = *(v11 - 1);
  }
  while ( *(v11 - 1) );
  v12 = a1;
  dword_4CF7F40 += 8;
  v13 = ");";
  sub_5D6390(v12);
  v14 = 41;
  do
  {
    ++v13;
    result = putc(v14, stream);
    v14 = *(v13 - 1);
  }
  while ( *(v13 - 1) );
  dword_4CF7F40 += 3;
  if ( stream != v3 )
    return sub_5D3B20(v3);
  return result;
}
