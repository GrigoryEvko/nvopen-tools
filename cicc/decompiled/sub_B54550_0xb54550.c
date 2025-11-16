// Function: sub_B54550
// Address: 0xb54550
//
__int64 __fastcall sub_B54550(__int64 a1, int a2)
{
  __int64 v2; // rax
  unsigned __int8 v3; // cl
  unsigned int v4; // edx
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // [rsp+8h] [rbp-18h]

  v2 = sub_BC89C0(a1);
  if ( !v2 )
    goto LABEL_4;
  v3 = *(_BYTE *)(v2 - 16);
  v4 = ((*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) >> 1) + 1;
  if ( (v3 & 2) != 0 )
  {
    if ( v4 != *(_DWORD *)(v2 - 24) )
    {
LABEL_4:
      BYTE4(v10) = 0;
      return v10;
    }
    v7 = *(_QWORD *)(v2 - 32);
    v6 = (unsigned int)(a2 + 1);
  }
  else
  {
    if ( v4 != ((*(_WORD *)(v2 - 16) >> 6) & 0xF) )
      goto LABEL_4;
    v6 = (unsigned int)(a2 + 1);
    v7 = v2 - 8LL * ((v3 >> 2) & 0xF) - 16;
  }
  v8 = *(_QWORD *)(*(_QWORD *)(v7 + 8 * v6) + 136LL);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  LODWORD(v10) = (_DWORD)v9;
  BYTE4(v10) = 1;
  return v10;
}
