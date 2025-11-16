// Function: sub_323E280
// Address: 0x323e280
//
void __fastcall sub_323E280(__int64 a1)
{
  _BYTE *v1; // rsi
  __int64 v2; // r15
  __int64 *v3; // r13
  __int64 v4; // rax
  __int64 v5; // r15
  unsigned __int8 v6; // dl
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  _BYTE **v10; // rbx
  __int64 v11; // rax
  _BYTE **v12; // r12
  _BYTE **v13; // rbx
  _BYTE **v14; // r12
  __int64 v15; // rax
  _UNKNOWN **v16; // rdx
  unsigned __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  int v20; // eax
  __int64 *v21; // [rsp+8h] [rbp-38h]

  v1 = *(_BYTE **)(a1 + 3056);
  if ( v1 )
    sub_3226DF0(a1, (__int64)v1);
  v2 = *(unsigned int *)(a1 + 664);
  v3 = *(__int64 **)(a1 + 656);
  *(_QWORD *)(a1 + 3056) = 0;
  v2 *= 16;
  v21 = (__int64 *)((char *)v3 + v2);
  if ( v3 != (__int64 *)((char *)v3 + v2) )
  {
    while ( 1 )
    {
      v4 = *v3;
      v5 = v3[1];
      v6 = *(_BYTE *)(*v3 - 16);
      if ( (v6 & 2) != 0 )
        v7 = *(_QWORD *)(v4 - 32);
      else
        v7 = v4 - 16 - 8LL * ((v6 >> 2) & 0xF);
      v8 = *(_QWORD *)(v7 + 56);
      if ( v8 )
      {
        v9 = *(_BYTE *)(v8 - 16);
        if ( (v9 & 2) != 0 )
        {
          v10 = *(_BYTE ***)(v8 - 32);
          v11 = *(unsigned int *)(v8 - 24);
        }
        else
        {
          v10 = (_BYTE **)(v8 - 16 - 8LL * ((v9 >> 2) & 0xF));
          v11 = (*(_WORD *)(v8 - 16) >> 6) & 0xF;
        }
        v12 = &v10[v11];
        while ( v12 != v10 )
        {
          v1 = *v10++;
          sub_37409E0(v5, v1);
        }
      }
      v13 = *(_BYTE ***)(v5 + 592);
      v14 = &v13[*(unsigned int *)(v5 + 600)];
      if ( v14 != v13 )
        break;
LABEL_14:
      v3 += 2;
      sub_373A8D0(v5);
      if ( v21 == v3 )
        goto LABEL_15;
    }
    while ( 1 )
    {
      v1 = *v13;
      if ( **v13 != 29 )
        break;
      ++v13;
      sub_37409E0(v5, v1);
      if ( v14 == v13 )
        goto LABEL_14;
    }
LABEL_37:
    BUG();
  }
LABEL_15:
  v15 = *(_QWORD *)(a1 + 8);
  if ( v15 && *(_BYTE *)(v15 + 782) )
  {
    sub_3239180(a1);
    if ( *(_BYTE *)(a1 + 3769) )
      sub_323CFC0(a1);
    else
      sub_323CF60(a1);
    sub_321F830(a1);
    sub_321F800(a1);
    if ( *(_BYTE *)(a1 + 3690) )
      sub_322FCD0((_QWORD *)a1, (__int64)v1, v16, v17, v18, v19);
    sub_323E1C0(a1);
    if ( *(_BYTE *)(a1 + 3769) )
      sub_3225810(a1);
    else
      sub_32257D0(a1);
    sub_321F9B0(a1);
    if ( *(_BYTE *)(a1 + 3769) )
    {
      sub_32209A0(a1);
      sub_32208C0(a1);
      sub_32208E0(a1);
      sub_3220910(a1);
      sub_323E240(a1);
    }
    sub_3220A00(a1);
    v20 = *(_DWORD *)(a1 + 3764);
    if ( v20 == 3 )
    {
      sub_321F8D0(a1);
    }
    else if ( v20 <= 3 )
    {
      if ( !v20 )
        goto LABEL_37;
      if ( v20 == 2 )
      {
        sub_3223920(a1);
        sub_3223960(a1);
        sub_32239A0(a1);
        sub_321F900(a1);
      }
    }
    sub_32228B0(a1);
  }
}
