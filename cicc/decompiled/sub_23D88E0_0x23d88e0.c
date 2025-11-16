// Function: sub_23D88E0
// Address: 0x23d88e0
//
__int64 __fastcall sub_23D88E0(__int64 a1, unsigned __int64 a2, __int64 *a3)
{
  __int64 v6; // rcx
  int v7; // edx
  _BYTE *v8; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // r8
  int v15; // edx
  int v16; // r9d
  __int64 v17; // [rsp+8h] [rbp-18h]

  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(unsigned __int8 *)(v6 + 8);
  if ( (unsigned int)(v7 - 17) <= 1 )
  {
    BYTE4(v17) = (_BYTE)v7 == 18;
    LODWORD(v17) = *(_DWORD *)(v6 + 32);
    a3 = (__int64 *)sub_BCE1B0(a3, v17);
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    v8 = (_BYTE *)sub_AD4C30(a2, (__int64 **)a3, 0);
    return sub_97B670(v8, *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 8));
  }
  v10 = *(unsigned int *)(a1 + 112);
  v11 = *(_QWORD *)(a1 + 96);
  if ( (_DWORD)v10 )
  {
    v12 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = (__int64 *)(v11 + 16LL * v12);
    v14 = *v13;
    if ( a2 == *v13 )
    {
LABEL_7:
      if ( v13 != (__int64 *)(v11 + 16 * v10) )
        return *(_QWORD *)(*(_QWORD *)(a1 + 120) + 24LL * *((unsigned int *)v13 + 2) + 16);
    }
    else
    {
      v15 = 1;
      while ( v14 != -4096 )
      {
        v16 = v15 + 1;
        v12 = (v10 - 1) & (v15 + v12);
        v13 = (__int64 *)(v11 + 16LL * v12);
        v14 = *v13;
        if ( a2 == *v13 )
          goto LABEL_7;
        v15 = v16;
      }
    }
  }
  return 0;
}
