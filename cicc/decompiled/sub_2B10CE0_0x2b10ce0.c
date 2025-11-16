// Function: sub_2B10CE0
// Address: 0x2b10ce0
//
__int64 __fastcall sub_2B10CE0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rax
  __int64 *v3; // rbx
  __int64 v4; // r15
  unsigned int v6; // esi
  __int64 v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdi
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // rax
  _BYTE *v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx

  v1 = **(_QWORD **)a1;
  v2 = **(_QWORD **)(a1 + 8);
  v3 = *(__int64 **)v2;
  v4 = *(_QWORD *)v2 + 8LL * *(unsigned int *)(v2 + 8);
  if ( *(_QWORD *)v2 != v4 )
  {
    do
    {
      while ( 1 )
      {
        v13 = (_BYTE *)*v3;
        if ( *(_BYTE *)*v3 > 0x1Cu )
          break;
LABEL_11:
        if ( (__int64 *)v4 == ++v3 )
          return v1;
      }
      v14 = *(_QWORD *)(v1 + 40);
      v15 = *((_QWORD *)v13 + 5);
      if ( v14 == v15 )
      {
        if ( sub_B445A0(*v3, v1) )
          v1 = (__int64)v13;
        goto LABEL_11;
      }
      v16 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 3320LL);
      if ( v14 )
      {
        v6 = *(_DWORD *)(v16 + 32);
        v7 = (unsigned int)(*(_DWORD *)(v14 + 44) + 1);
        if ( *(_DWORD *)(v14 + 44) + 1 >= v6 )
          goto LABEL_16;
      }
      else
      {
        v6 = *(_DWORD *)(v16 + 32);
        v7 = 0;
        if ( !v6 )
          goto LABEL_16;
      }
      v8 = *(_QWORD *)(v16 + 24);
      v9 = *(_QWORD *)(v8 + 8 * v7);
      if ( v9 )
      {
        if ( v15 )
        {
          v10 = (unsigned int)(*(_DWORD *)(v15 + 44) + 1);
          v11 = v10;
        }
        else
        {
          v10 = 0;
          v11 = 0;
        }
        if ( v6 > v11 )
        {
          v12 = *(_QWORD *)(v8 + 8 * v10);
          if ( v12 )
          {
            if ( *(_DWORD *)(v9 + 72) > *(_DWORD *)(v12 + 72) )
              v1 = *v3;
          }
        }
        goto LABEL_11;
      }
LABEL_16:
      ++v3;
      v1 = (__int64)v13;
    }
    while ( (__int64 *)v4 != v3 );
  }
  return v1;
}
