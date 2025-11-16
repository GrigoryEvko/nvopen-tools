// Function: sub_27FF330
// Address: 0x27ff330
//
void __fastcall sub_27FF330(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // rsi
  __int64 *v10; // r14
  __int64 *v11; // rax
  __int64 *v12; // r12
  __int64 v13; // r12
  __int64 v14; // rax
  unsigned int v15; // edi
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // rax

  *(_QWORD *)(a3 + 8) = a2;
  if ( !a2 )
    return;
  v5 = *(_QWORD *)(a2 - 64);
  if ( !v5 )
    return;
  *(_QWORD *)(a3 + 24) = v5;
  v6 = *(_QWORD *)(a2 - 32);
  if ( !v6 )
    return;
  *(_QWORD *)(a3 + 40) = v6;
  v8 = sub_B53900(a2);
  v9 = *(_QWORD *)(a3 + 24);
  *(_DWORD *)(a3 + 16) = v8;
  *(_BYTE *)(a3 + 20) = BYTE4(v8);
  v10 = sub_DD8400(a1, v9);
  v11 = sub_DD8400(a1, *(_QWORD *)(a3 + 40));
  v12 = v11;
  if ( *((_WORD *)v10 + 12) != 8 )
  {
    if ( *((_WORD *)v11 + 12) == 8 )
    {
      v14 = *(_QWORD *)(a3 + 24);
      v15 = *(_DWORD *)(a3 + 16);
      *(_QWORD *)(a3 + 24) = *(_QWORD *)(a3 + 40);
      *(_QWORD *)(a3 + 40) = v14;
      v16 = sub_B52F50(v15);
      *(_BYTE *)(a3 + 20) = 0;
      *(_DWORD *)(a3 + 16) = v16;
      if ( *((_WORD *)v12 + 12) == 8 )
      {
        v23 = v10;
        v10 = v12;
        v12 = v23;
        goto LABEL_5;
      }
    }
    else
    {
      v10 = v11;
    }
    v17 = *(_QWORD *)(a3 + 24);
    *(_QWORD *)(a3 + 56) = v10;
    *(_QWORD *)(a3 + 48) = 0;
    *(_QWORD *)(a3 + 32) = v17;
    return;
  }
LABEL_5:
  *(_QWORD *)(a3 + 56) = v12;
  v13 = *(_QWORD *)(a3 + 24);
  *(_QWORD *)(a3 + 48) = v10;
  *(_QWORD *)(a3 + 32) = v13;
  if ( *(_BYTE *)v13 == 84 )
  {
    v18 = sub_D47930(a4);
    v19 = *(_QWORD *)(v13 - 8);
    v20 = v18;
    if ( (*(_DWORD *)(v13 + 4) & 0x7FFFFFF) != 0 )
    {
      v21 = 0;
      while ( v20 != *(_QWORD *)(v19 + 32LL * *(unsigned int *)(v13 + 72) + 8 * v21) )
      {
        if ( (*(_DWORD *)(v13 + 4) & 0x7FFFFFF) == (_DWORD)++v21 )
          goto LABEL_17;
      }
      v22 = 32 * v21;
    }
    else
    {
LABEL_17:
      v22 = 0x1FFFFFFFE0LL;
    }
    *(_QWORD *)(a3 + 32) = *(_QWORD *)(v19 + v22);
  }
}
