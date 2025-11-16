// Function: sub_C68D40
// Address: 0xc68d40
//
__int64 __fastcall sub_C68D40(__int64 *a1, __int64 a2)
{
  __int64 *v3; // r14
  unsigned __int64 v4; // rsi
  char *v5; // rcx
  char *v6; // rax
  int v7; // r12d
  char v8; // dl
  int v9; // r13d
  int v10; // ebx
  __int64 v11; // rax
  __int64 v13; // rdx

  v3 = a1;
  v4 = a1[2];
  v5 = (char *)a1[3];
  if ( (unsigned __int64)v5 <= v4 )
  {
    a1 = (__int64 *)a1[2];
    v7 = 1;
  }
  else
  {
    v6 = (char *)a1[2];
    LODWORD(a1) = (_DWORD)v6;
    v7 = 1;
    do
    {
      v8 = *v6++;
      if ( v8 == 10 )
      {
        ++v7;
        LODWORD(a1) = (_DWORD)v6;
      }
    }
    while ( v6 != v5 );
  }
  v9 = (_DWORD)v5 - (_DWORD)a1;
  v10 = (_DWORD)v5 - v4;
  v11 = sub_22077B0(32);
  if ( v11 )
  {
    *(_QWORD *)(v11 + 8) = a2;
    *(_DWORD *)(v11 + 16) = v7;
    *(_DWORD *)(v11 + 20) = v9;
    *(_QWORD *)v11 = &unk_49DC8C0;
    *(_DWORD *)(v11 + 24) = v10;
  }
  if ( *((_BYTE *)v3 + 8) )
  {
    v13 = *v3;
    *((_BYTE *)v3 + 8) = 0;
    if ( (v13 & 1) != 0 || (v13 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(v3, v4);
  }
  *((_BYTE *)v3 + 8) = 1;
  *v3 = v11 | 1;
  return 0;
}
