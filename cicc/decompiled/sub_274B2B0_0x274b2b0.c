// Function: sub_274B2B0
// Address: 0x274b2b0
//
const void *__fastcall sub_274B2B0(__int64 *a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5)
{
  const void *v8; // r12
  __int64 v10; // r9
  __int64 v11; // rsi
  _BYTE *v12; // rcx
  __int64 v13; // rax
  int v14; // eax
  bool v15; // al
  _BYTE *v16; // rcx
  __int64 v17; // rax
  unsigned int v18; // r13d
  _BYTE *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  bool v27; // al
  __int64 v28; // [rsp+0h] [rbp-40h]
  __int64 v29; // [rsp+0h] [rbp-40h]
  __int64 v31; // [rsp+8h] [rbp-38h]
  int v32; // [rsp+8h] [rbp-38h]
  __int64 v33; // [rsp+8h] [rbp-38h]

  v8 = sub_22CF3A0(a1, (__int64)a2, a3, a4, a5);
  if ( !v8 )
  {
    v10 = a4;
    if ( *a2 == 86 )
    {
      v11 = *((_QWORD *)a2 - 12);
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v11 + 8) + 8LL) - 17 > 1 )
      {
        v20 = sub_22CF3A0(a1, v11, a3, a4, a5);
        v10 = a4;
        if ( v20 )
        {
          v29 = a4;
          v33 = (__int64)v20;
          if ( sub_AD7A80(v20, v11, v21, v22, v23) )
            return (const void *)*((_QWORD *)a2 - 8);
          v27 = sub_AD7890(v33, v11, v24, v25, v26);
          v10 = v29;
          if ( v27 )
            return (const void *)*((_QWORD *)a2 - 4);
        }
      }
      v12 = (_BYTE *)*((_QWORD *)a2 - 4);
      if ( *v12 <= 0x15u )
      {
        v31 = v10;
        v13 = sub_22CF6C0(a1, 0x20u, (__int64)a2, (__int64)v12, a3, v10, a5);
        v10 = v31;
        if ( v13 )
        {
          if ( *(_BYTE *)v13 == 17 )
          {
            if ( *(_DWORD *)(v13 + 32) <= 0x40u )
            {
              v15 = *(_QWORD *)(v13 + 24) == 0;
            }
            else
            {
              v28 = v31;
              v32 = *(_DWORD *)(v13 + 32);
              v14 = sub_C444A0(v13 + 24);
              v10 = v28;
              v15 = v32 == v14;
            }
            if ( v15 )
              return (const void *)*((_QWORD *)a2 - 8);
          }
        }
      }
      v16 = (_BYTE *)*((_QWORD *)a2 - 8);
      if ( *v16 <= 0x15u )
      {
        v17 = sub_22CF6C0(a1, 0x20u, (__int64)a2, (__int64)v16, a3, v10, a5);
        if ( v17 )
        {
          if ( *(_BYTE *)v17 == 17 )
          {
            v18 = *(_DWORD *)(v17 + 32);
            if ( v18 <= 0x40 ? *(_QWORD *)(v17 + 24) == 0 : v18 == (unsigned int)sub_C444A0(v17 + 24) )
              return (const void *)*((_QWORD *)a2 - 4);
          }
        }
      }
    }
  }
  return v8;
}
