// Function: sub_349D6E0
// Address: 0x349d6e0
//
__int64 __fastcall sub_349D6E0(__int64 a1, unsigned __int64 a2)
{
  char v3; // cl
  __int64 v4; // r10
  int v5; // edi
  unsigned int v6; // r8d
  int *v7; // rax
  int v8; // edx
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // eax
  int v13; // ebx

  v3 = *(_BYTE *)(a1 + 56) & 1;
  if ( v3 )
  {
    v4 = a1 + 64;
    v5 = 3;
  }
  else
  {
    v10 = *(unsigned int *)(a1 + 72);
    v4 = *(_QWORD *)(a1 + 64);
    if ( !(_DWORD)v10 )
      goto LABEL_8;
    v5 = v10 - 1;
  }
  v6 = v5 & (37 * a2);
  v7 = (int *)(v4 + 32LL * v6);
  v8 = *v7;
  if ( (_DWORD)a2 != *v7 )
  {
    v12 = 1;
    while ( v8 != -1 )
    {
      v13 = v12 + 1;
      v6 = v5 & (v12 + v6);
      v7 = (int *)(v4 + 32LL * v6);
      v8 = *v7;
      if ( (_DWORD)a2 == *v7 )
        return *((_QWORD *)v7 + 1) + 384 * HIDWORD(a2);
      v12 = v13;
    }
    if ( v3 )
    {
      v11 = 128;
      goto LABEL_9;
    }
    v10 = *(unsigned int *)(a1 + 72);
LABEL_8:
    v11 = 32 * v10;
LABEL_9:
    v7 = (int *)(v4 + v11);
  }
  return *((_QWORD *)v7 + 1) + 384 * HIDWORD(a2);
}
