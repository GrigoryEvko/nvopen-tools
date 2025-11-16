// Function: sub_2950400
// Address: 0x2950400
//
__int64 __fastcall sub_2950400(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int8 *v4; // r15
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // r10
  unsigned __int8 *v8; // r14
  int v9; // eax
  int v10; // edi
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // [rsp-78h] [rbp-78h]
  int v17; // [rsp-70h] [rbp-70h]
  _BYTE v18[32]; // [rsp-68h] [rbp-68h] BYREF
  __int16 v19; // [rsp-48h] [rbp-48h]

  v2 = *a1;
  if ( (_DWORD)a2 )
  {
    v4 = *(unsigned __int8 **)(v2 + 8LL * (unsigned int)a2);
    v5 = *(_QWORD *)(v2 + 8LL * (unsigned int)(a2 - 1));
    v6 = *((_QWORD *)v4 - 8);
    v7 = sub_2950400(a1, (unsigned int)(a2 - 1));
    v8 = *(unsigned __int8 **)&v4[32 * (v5 == v6) - 64];
    if ( *(_BYTE *)v7 != 17 )
      goto LABEL_5;
    if ( *(_DWORD *)(v7 + 32) <= 0x40u )
    {
      if ( *(_QWORD *)(v7 + 24) )
      {
LABEL_5:
        v10 = *v4 - 29;
        if ( *v4 == 58 )
          v10 = 13;
        if ( v5 != v6 )
        {
          v11 = a1[28];
          v12 = *((unsigned __int16 *)a1 + 116);
          v19 = 257;
          v8 = (unsigned __int8 *)sub_B504D0(v10, (__int64)v8, v7, (__int64)v18, v11, v12);
LABEL_9:
          sub_BD6B90(v8, v4);
          return (__int64)v8;
        }
LABEL_15:
        v14 = a1[28];
        v15 = *((unsigned __int16 *)a1 + 116);
        v19 = 257;
        v8 = (unsigned __int8 *)sub_B504D0(v10, v7, (__int64)v8, (__int64)v18, v14, v15);
        goto LABEL_9;
      }
    }
    else
    {
      v17 = *(_DWORD *)(v7 + 32);
      v16 = v7;
      v9 = sub_C444A0(v7 + 24);
      v7 = v16;
      if ( v17 != v9 )
        goto LABEL_5;
    }
    if ( *v4 != 44 )
      return (__int64)v8;
    v10 = 15;
    if ( v5 != v6 )
      return (__int64)v8;
    goto LABEL_15;
  }
  return sub_AD6530(*(_QWORD *)(*(_QWORD *)v2 + 8LL), a2);
}
