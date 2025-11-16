// Function: sub_3577E50
// Address: 0x3577e50
//
__int64 __fastcall sub_3577E50(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 (*v3)(void); // rax
  __int64 *v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _BYTE *v9; // r15
  _BYTE *v10; // r12
  _BYTE *v11; // rbx
  int v12; // esi
  __int64 v13; // rax
  __int64 (*v14)(); // rdx
  __int64 (*v15)(); // rax
  _BYTE *v16; // r15
  __int64 v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h]
  unsigned __int8 v20; // [rsp+1Fh] [rbp-31h]

  v2 = *(_QWORD *)(a1 + 216);
  v18 = 0;
  v19 = *(_QWORD *)(v2 + 32);
  v3 = *(__int64 (**)(void))(**(_QWORD **)(v2 + 16) + 208LL);
  if ( v3 != sub_2EEE460 )
    v18 = v3();
  v4 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)v19 + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)v19 + 16LL));
  if ( !sub_2E5E6D0(*(_QWORD *)(a1 + 224), *(_QWORD *)(a2 + 24))
    || (v20 = 0, !(unsigned __int8)sub_3574610(a1, a2, v5, v6, v7, v8)) )
  {
    v9 = *(_BYTE **)(a2 + 32);
    v10 = &v9[40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF)];
    if ( v9 == v10 )
      return 0;
    while ( 1 )
    {
      v11 = v9;
      if ( sub_2DADC00(v9) )
        break;
      v9 += 40;
      if ( v10 == v9 )
        return 0;
    }
    v20 = 0;
    if ( v10 != v9 )
    {
      while ( 1 )
      {
        v12 = *((_DWORD *)v11 + 2);
        v13 = *v4;
        if ( v12 >= 0 )
        {
          v14 = *(__int64 (**)())(v13 + 168);
          if ( v14 != sub_2EA3FB0 )
          {
            if ( ((unsigned __int8 (__fastcall *)(__int64 *))v14)(v4) )
              goto LABEL_14;
            v13 = *v4;
            v12 = *((_DWORD *)v11 + 2);
          }
        }
        v15 = *(__int64 (**)())(v13 + 184);
        if ( v15 == sub_2FF5200 )
          goto LABEL_13;
        if ( !((unsigned __int8 (__fastcall *)(__int64 *, __int64, __int64, _QWORD))v15)(
                v4,
                v19,
                v18,
                (unsigned int)v12) )
        {
          v12 = *((_DWORD *)v11 + 2);
LABEL_13:
          v20 |= sub_3577C00(a1, v12);
        }
LABEL_14:
        if ( v11 + 40 != v10 )
        {
          v16 = v11 + 40;
          while ( 1 )
          {
            v11 = v16;
            if ( sub_2DADC00(v16) )
              break;
            v16 += 40;
            if ( v10 == v16 )
              return v20;
          }
          if ( v10 != v16 )
            continue;
        }
        return v20;
      }
    }
  }
  return v20;
}
