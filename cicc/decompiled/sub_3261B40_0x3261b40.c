// Function: sub_3261B40
// Address: 0x3261b40
//
__int64 __fastcall sub_3261B40(__int16 **a1, __int64 a2)
{
  _QWORD *v4; // rbx
  __int64 v5; // rsi
  _QWORD *v6; // r12
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int16 v11; // ax
  __int64 v12; // rdi
  __int64 *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rax
  int v17; // edx
  int v18; // ecx
  __int64 v19; // rdi
  __int128 v21; // [rsp-10h] [rbp-80h]
  int v22; // [rsp+0h] [rbp-70h]
  __int64 v23; // [rsp+8h] [rbp-68h]
  __int64 v24; // [rsp+30h] [rbp-40h] BYREF
  int v25; // [rsp+38h] [rbp-38h]

  v4 = *(_QWORD **)a2;
  v5 = *(unsigned int *)(a2 + 8);
  v6 = &v4[2 * v5];
  if ( v6 != v4 )
  {
    while ( 1 )
    {
      while ( !*v4 )
      {
        v12 = *(_QWORD *)a1[1];
        v13 = (__int64 *)a1[3];
        v14 = *v13;
        v15 = v13[1];
        v24 = 0;
        v25 = 0;
        v16 = sub_33F17F0(v12, 51, &v24, v14, v15);
        v18 = v17;
        v19 = v16;
        if ( v24 )
        {
          v22 = v17;
          v23 = v16;
          sub_B91220((__int64)&v24, v24);
          v18 = v22;
          v19 = v23;
        }
        v4 += 2;
        *(v4 - 2) = v19;
        *((_DWORD *)v4 - 2) = v18;
        if ( v6 == v4 )
        {
LABEL_15:
          v4 = *(_QWORD **)a2;
          v5 = *(unsigned int *)(a2 + 8);
          goto LABEL_16;
        }
      }
      v11 = **a1;
      if ( v11 )
      {
        if ( (unsigned __int16)(v11 - 2) > 7u
          && (unsigned __int16)(v11 - 17) > 0x6Cu
          && (unsigned __int16)(v11 - 176) > 0x1Fu )
        {
LABEL_11:
          v10 = *v4;
          v9 = v4[1];
          goto LABEL_7;
        }
      }
      else if ( !sub_3007070((__int64)*a1) )
      {
        goto LABEL_11;
      }
      v7 = sub_33FAFB0(*(_QWORD *)a1[1], *v4, v4[1], a1[2], *(unsigned int *)a1[3], *((_QWORD *)a1[3] + 1));
      LODWORD(v9) = v8;
      v10 = v7;
LABEL_7:
      v4 += 2;
      *(v4 - 2) = v10;
      *((_DWORD *)v4 - 2) = v9;
      if ( v6 == v4 )
        goto LABEL_15;
    }
  }
LABEL_16:
  *((_QWORD *)&v21 + 1) = v5;
  *(_QWORD *)&v21 = v4;
  return sub_33FC220(
           *(_QWORD *)a1[1],
           156,
           (unsigned int)a1[2],
           *(_QWORD *)*a1,
           *((_QWORD *)*a1 + 1),
           (unsigned int)a1[2],
           v21);
}
