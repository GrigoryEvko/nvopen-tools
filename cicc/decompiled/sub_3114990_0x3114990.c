// Function: sub_3114990
// Address: 0x3114990
//
__int64 __fastcall sub_3114990(unsigned __int64 a1, __int64 a2)
{
  unsigned __int64 *v3; // r14
  __int64 v4; // rbx
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rdi
  _QWORD *v7; // r8
  _QWORD *v8; // rax
  _QWORD *v9; // rsi
  __int64 result; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // r13
  __int64 v13; // rax
  _QWORD *v14; // rax
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // r8
  _QWORD *v17; // r11
  unsigned __int64 v18; // r10
  _QWORD *v19; // rax
  _QWORD *v20; // rdi
  unsigned __int64 v21; // [rsp+0h] [rbp-40h]

  v3 = *(unsigned __int64 **)a2;
  v4 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v4 )
  {
    while ( 1 )
    {
      v5 = *v3;
      v6 = *(_QWORD *)(a1 + 24);
      v7 = *(_QWORD **)(*(_QWORD *)(a1 + 16) + 8 * (*v3 % v6));
      if ( !v7 )
        goto LABEL_15;
      v8 = (_QWORD *)*v7;
      if ( v5 != *(_QWORD *)(*v7 + 8LL) )
        break;
LABEL_7:
      if ( !*v7 )
        goto LABEL_15;
      a1 = *(_QWORD *)(*v7 + 16LL);
LABEL_9:
      if ( (unsigned __int64 *)v4 == ++v3 )
        goto LABEL_10;
    }
    while ( 1 )
    {
      v9 = (_QWORD *)*v8;
      if ( !*v8 )
        break;
      v7 = v8;
      if ( *v3 % v6 != v9[1] % v6 )
        break;
      v8 = (_QWORD *)*v8;
      if ( v5 == v9[1] )
        goto LABEL_7;
    }
LABEL_15:
    v11 = sub_22077B0(0x48u);
    v12 = v11;
    if ( v11 )
    {
      *(_QWORD *)(v11 + 64) = 0;
      v13 = v11 + 64;
      *(_OWORD *)(v13 - 64) = 0;
      *(_OWORD *)(v13 - 32) = 0;
      *(_OWORD *)(v13 - 16) = 0;
      *(_QWORD *)(v12 + 16) = v13;
      *(_QWORD *)(v12 + 24) = 1;
      *(_DWORD *)(v12 + 48) = 1065353216;
      *(_QWORD *)(v12 + 56) = 0;
    }
    *(_QWORD *)v12 = v5;
    v14 = (_QWORD *)sub_22077B0(0x18u);
    v15 = (unsigned __int64)v14;
    if ( v14 )
      *v14 = 0;
    v14[1] = v5;
    v14[2] = v12;
    v16 = *(_QWORD *)(a1 + 24);
    v17 = *(_QWORD **)(*(_QWORD *)(a1 + 16) + 8 * (v5 % v16));
    v18 = v5 % v16;
    if ( v17 )
    {
      v19 = (_QWORD *)*v17;
      if ( v5 == *(_QWORD *)(*v17 + 8LL) )
      {
LABEL_24:
        if ( *v17 )
        {
          v21 = v15;
          sub_31142D0(v12);
          j_j___libc_free_0(v21);
LABEL_26:
          a1 = v12;
          goto LABEL_9;
        }
      }
      else
      {
        while ( 1 )
        {
          v20 = (_QWORD *)*v19;
          if ( !*v19 )
            break;
          v17 = v19;
          if ( v18 != v20[1] % v16 )
            break;
          v19 = (_QWORD *)*v19;
          if ( v5 == v20[1] )
            goto LABEL_24;
        }
      }
    }
    sub_3113D80((unsigned __int64 *)(a1 + 16), v18, v5, v15, 1);
    goto LABEL_26;
  }
LABEL_10:
  result = *(unsigned int *)(a2 + 64);
  if ( (_DWORD)result )
  {
    if ( *(_BYTE *)(a1 + 12) )
      result = (unsigned int)(*(_DWORD *)(a1 + 8) + result);
    *(_DWORD *)(a1 + 8) = result;
    *(_BYTE *)(a1 + 12) = 1;
  }
  return result;
}
