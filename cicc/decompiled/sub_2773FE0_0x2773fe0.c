// Function: sub_2773FE0
// Address: 0x2773fe0
//
__int64 __fastcall sub_2773FE0(__int64 a1, char *a2, _QWORD *a3)
{
  int v3; // eax
  int v5; // r11d
  __int64 v6; // r12
  __int64 v7; // r9
  int v8; // ebx
  __int64 v9; // r10
  __int64 v10; // rdi
  unsigned __int8 v11; // si
  unsigned int i; // ecx
  unsigned __int8 *v13; // rax
  unsigned int v14; // r8d
  __int64 v15; // r13
  unsigned int v16; // ecx

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = v3 - 1;
  v6 = 0;
  v7 = *((_QWORD *)a2 + 6);
  v8 = 1;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *((_QWORD *)a2 + 3);
  v11 = *a2;
  for ( i = (v3 - 1) & (v11 ^ v10 ^ v7); ; i = v5 & v16 )
  {
    v13 = (unsigned __int8 *)(v9 + ((unsigned __int64)i << 6));
    v14 = *v13;
    v15 = *((_QWORD *)v13 + 3);
    if ( v11 == (_BYTE)v14 && v10 == v15 && v7 == *((_QWORD *)v13 + 6) )
    {
      *a3 = v13;
      return 1;
    }
    if ( !(_BYTE)v14 )
      break;
    if ( !v15 && !(*((_QWORD *)v13 + 6) | v6) )
      v6 = v9 + ((unsigned __int64)i << 6);
LABEL_7:
    v16 = v8 + i;
    ++v8;
  }
  if ( v15 || *((_QWORD *)v13 + 6) )
    goto LABEL_7;
  if ( !v6 )
    v6 = v9 + ((unsigned __int64)i << 6);
  *a3 = v6;
  return v14;
}
