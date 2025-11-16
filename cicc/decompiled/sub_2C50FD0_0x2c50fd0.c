// Function: sub_2C50FD0
// Address: 0x2c50fd0
//
unsigned __int64 __fastcall sub_2C50FD0(__int64 a1, unsigned int a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int8 *v11; // r8
  unsigned int v12; // eax
  unsigned int v13; // r10d
  int v14; // esi
  unsigned __int8 *v15; // rcx
  int v16; // edx
  int v17; // edx
  __int64 v19; // rax
  unsigned int v20; // edx
  unsigned int v21; // [rsp+1Ch] [rbp-84h]
  __int64 v22; // [rsp+20h] [rbp-80h]
  unsigned __int8 *v23; // [rsp+28h] [rbp-78h]
  __int64 v24; // [rsp+68h] [rbp-38h]

  v9 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v10 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v9 <= 0x1Cu )
  {
    if ( v10 )
    {
      v11 = 0;
      goto LABEL_4;
    }
LABEL_18:
    BUG();
  }
  if ( !v10 )
    goto LABEL_18;
  v11 = *(unsigned __int8 **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
LABEL_4:
  v23 = v11;
  if ( *(_BYTE *)v10 || *(_QWORD *)(v10 + 24) != *(_QWORD *)(a1 + 80) )
    goto LABEL_19;
  v22 = *(_QWORD *)(v9 + 8);
  v12 = sub_F6EED0(*(_DWORD *)(v10 + 36));
  v13 = v12;
  if ( !v23 )
  {
LABEL_15:
    BYTE4(v24) = 0;
    *(_QWORD *)a5 = sub_DFDC10(a3, v13, v22, v24);
    *(_DWORD *)(a5 + 8) = v20;
    return v20;
  }
  v14 = *v23;
  if ( (unsigned __int8)(v14 - 68) > 1u )
  {
    v19 = *(_QWORD *)(a1 - 32);
    if ( v19 && !*(_BYTE *)v19 && *(_QWORD *)(v19 + 24) == *(_QWORD *)(a1 + 80) )
      goto LABEL_15;
LABEL_19:
    BUG();
  }
  if ( (v23[7] & 0x40) != 0 )
    v15 = (unsigned __int8 *)*((_QWORD *)v23 - 1);
  else
    v15 = &v23[-32 * (*((_DWORD *)v23 + 1) & 0x7FFFFFF)];
  v21 = v12;
  *(_QWORD *)a4 = sub_DFD060(a3, (unsigned int)(v14 - 29), v22, *(_QWORD *)(*(_QWORD *)v15 + 8LL));
  *(_DWORD *)(a4 + 8) = v16;
  *(_QWORD *)a5 = sub_DFDCC0(a3, v21, (_BYTE)v14 == 68, *(_QWORD *)(a1 + 8));
  *(_DWORD *)(a5 + 8) = v17;
  return __PAIR64__(HIDWORD(v23), a2);
}
