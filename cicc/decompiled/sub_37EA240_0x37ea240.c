// Function: sub_37EA240
// Address: 0x37ea240
//
__int64 __fastcall sub_37EA240(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  __int64 v8; // r15
  __int64 result; // rax
  _QWORD *v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // edx
  __int16 *v14; // rsi
  _WORD *v15; // rax
  int v16; // edi
  __int64 v17; // rax
  __int64 v18; // r14
  _BYTE *v19; // rdx
  _BYTE *v20; // r14
  unsigned int v21; // esi
  __int64 v22; // rax
  int v23; // eax
  _BYTE *v24; // rdx
  __int64 v25; // r14
  unsigned __int16 *v26; // rcx
  int v27; // r14d
  unsigned int v28; // eax
  unsigned __int16 ***v29; // [rsp+8h] [rbp-58h]
  int v30; // [rsp+8h] [rbp-58h]
  unsigned __int16 **v31; // [rsp+10h] [rbp-50h]
  unsigned __int16 *v32; // [rsp+10h] [rbp-50h]
  unsigned int v33; // [rsp+1Ch] [rbp-44h]
  unsigned int v34; // [rsp+20h] [rbp-40h]
  _BYTE *v35; // [rsp+28h] [rbp-38h]
  unsigned __int8 v36; // [rsp+28h] [rbp-38h]
  unsigned __int16 *v37; // [rsp+28h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 32) + 40LL * a3;
  if ( *(_BYTE *)v8 || (*(_BYTE *)(v8 + 3) & 0x10) != 0 || (result = 0, (*(_WORD *)(v8 + 2) & 0xFF0) == 0) )
  {
    if ( !(unsigned __int8)sub_2EAB300(v8) )
      return 0;
    v10 = *(_QWORD **)(a1 + 216);
    v11 = v10[1];
    v33 = *(_DWORD *)(v8 + 8);
    v12 = *(_DWORD *)(v11 + 24LL * v33 + 16) >> 12;
    v13 = *(_DWORD *)(v11 + 24LL * v33 + 16) & 0xFFF;
    v14 = (__int16 *)(v10[7] + 2 * v12);
    do
    {
      if ( !v14 )
        break;
      v15 = (_WORD *)(v10[6] + 4LL * v13);
      if ( *v15 && v15[1] )
        return 0;
      v16 = *v14++;
      v13 += v16;
    }
    while ( (_WORD)v16 );
    v17 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD *, _QWORD))(**(_QWORD **)(a1 + 208) + 16LL))(
            *(_QWORD *)(a1 + 208),
            *(_QWORD *)(a2 + 16),
            a3,
            v10,
            *(_QWORD *)(a1 + 200));
    v18 = *(_QWORD *)(a2 + 32);
    v29 = (unsigned __int16 ***)v17;
    v35 = (_BYTE *)(v18 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF));
    v19 = (_BYTE *)(v18 + 40LL * (unsigned int)sub_2E88FE0(a2));
    if ( v35 == v19 )
    {
LABEL_36:
      v31 = *v29;
      goto LABEL_26;
    }
    while ( 1 )
    {
      v20 = v19;
      if ( (unsigned __int8)sub_2E2FA70(v19) )
        break;
      v19 = v20 + 40;
      if ( v35 == v20 + 40 )
        goto LABEL_36;
    }
    v31 = *v29;
    if ( v20 == v35 )
    {
LABEL_26:
      v25 = *(_QWORD *)(a1 + 224) + 24LL * *((unsigned __int16 *)v31 + 12);
      if ( *(_DWORD *)(a1 + 232) != *(_DWORD *)v25 )
        sub_2F60630(a1 + 224, v29);
      v26 = *(unsigned __int16 **)(v25 + 16);
      v32 = &v26[*(unsigned int *)(v25 + 4)];
      if ( v32 != v26 )
      {
        v34 = 0;
        v30 = v33;
        do
        {
          v27 = *v26;
          v37 = v26;
          v28 = sub_37F59F0(*(_QWORD *)(a1 + 632), a2, *v26);
          if ( v28 > v34 )
          {
            v30 = v27;
            if ( a4 < v28 )
              break;
            v34 = v28;
          }
          v26 = v37 + 1;
        }
        while ( v32 != v37 + 1 );
        if ( v33 != v30 )
        {
          sub_2EAB0C0(v8, v30);
          return 0;
        }
      }
      return 0;
    }
    while ( 1 )
    {
      if ( (v20[4] & 1) == 0 )
      {
        v21 = *((_DWORD *)v20 + 2);
        if ( v21 - 1 <= 0x3FFFFFFE )
        {
          v22 = v21 >> 3;
          if ( (unsigned int)v22 < *((unsigned __int16 *)v31 + 11) )
          {
            v23 = ((int)*((unsigned __int8 *)v31[1] + v22) >> (v21 & 7)) & 1;
            if ( v23 )
              break;
          }
        }
      }
      v24 = v20 + 40;
      if ( v20 + 40 != v35 )
      {
        while ( 1 )
        {
          v20 = v24;
          if ( (unsigned __int8)sub_2E2FA70(v24) )
            break;
          v24 = v20 + 40;
          if ( v35 == v20 + 40 )
            goto LABEL_26;
        }
        if ( v35 != v20 )
          continue;
      }
      goto LABEL_26;
    }
    v36 = v23;
    sub_2EAB0C0(v8, v21);
    return v36;
  }
  return result;
}
