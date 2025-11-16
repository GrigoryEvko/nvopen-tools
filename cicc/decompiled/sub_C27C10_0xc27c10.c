// Function: sub_C27C10
// Address: 0xc27c10
//
__int64 __fastcall sub_C27C10(_QWORD *a1, unsigned __int8 a2, __int64 a3)
{
  __int64 *v4; // rbx
  __int64 *v5; // r12
  unsigned int v6; // ecx
  __int64 v8; // r14
  unsigned __int64 v9; // r15
  unsigned __int64 v10; // rdi
  _QWORD *v11; // r10
  _QWORD *v12; // rax
  _QWORD *v13; // rsi
  _QWORD *v14; // rax
  __int64 v15; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v17; // [rsp+20h] [rbp-E0h] BYREF
  _BYTE v18[208]; // [rsp+30h] [rbp-D0h] BYREF

  if ( !a1[16] )
  {
    sub_C1AFD0();
    return 0;
  }
  v4 = *(__int64 **)(a3 + 8);
  v5 = &v4[*(unsigned int *)(a3 + 24)];
  if ( *(_DWORD *)(a3 + 16) && v5 != v4 )
  {
    while ( *v4 == -4096 || *v4 == -8192 )
    {
      if ( v5 == ++v4 )
        goto LABEL_4;
    }
    if ( v5 != v4 )
    {
      v8 = *v4;
      if ( *(_DWORD *)(*v4 + 48) )
      {
LABEL_13:
        v9 = sub_C1B290(*(__int64 **)(v8 + 32), (__int64 *)(*(_QWORD *)(v8 + 32) + 24LL * *(_QWORD *)(v8 + 40)));
      }
      else
      {
LABEL_26:
        v9 = *(_QWORD *)(v8 + 24);
        v15 = *(_QWORD *)(v8 + 16);
        if ( v15 )
        {
          sub_C7D030(v18);
          sub_C7D280(v18, v15, v9);
          sub_C7D290(v18, &v17);
          v9 = v17;
        }
      }
      v10 = a1[14];
      v11 = *(_QWORD **)(a1[13] + 8 * (v9 % v10));
      if ( !v11 )
        goto LABEL_21;
      v12 = (_QWORD *)*v11;
      if ( v9 == *(_QWORD *)(*v11 + 8LL) )
      {
LABEL_19:
        v14 = (_QWORD *)*v11;
        if ( !*v11 )
          goto LABEL_21;
        a1[26] = v14[2];
        a1[27] = v14[3];
        v6 = sub_C27850((__int64)a1, a2, v8);
        if ( !v6 )
          goto LABEL_21;
        return v6;
      }
      while ( 1 )
      {
        v13 = (_QWORD *)*v12;
        if ( !*v12 )
          break;
        v11 = v12;
        if ( v9 % v10 != v13[1] % v10 )
          break;
        v12 = (_QWORD *)*v12;
        if ( v9 == v13[1] )
          goto LABEL_19;
      }
LABEL_21:
      while ( v5 != ++v4 )
      {
        if ( *v4 != -8192 && *v4 != -4096 )
        {
          if ( v5 == v4 )
            break;
          v8 = *v4;
          if ( *(_DWORD *)(*v4 + 48) )
            goto LABEL_13;
          goto LABEL_26;
        }
      }
    }
  }
LABEL_4:
  sub_C1AFD0();
  return 0;
}
