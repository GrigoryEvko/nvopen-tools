// Function: sub_3353030
// Address: 0x3353030
//
__int64 __fastcall sub_3353030(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  int v7; // eax
  __int64 v8; // rdx
  unsigned int v9; // esi
  unsigned __int16 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // r11
  __int64 v13; // rcx
  unsigned int v14; // r12d
  __int64 v15; // rcx
  unsigned __int64 v16; // rdx
  unsigned __int16 *v17; // rax
  unsigned __int16 *v18; // r14
  __int64 v19; // r9
  __int64 v20; // rax
  unsigned int v21; // r13d
  unsigned __int16 *v22; // r15
  __int64 v23; // r14
  __int16 v24; // ax
  unsigned int v25; // r11d
  unsigned int v26; // esi
  int v27; // eax
  unsigned __int16 *v28; // rbx
  unsigned int v29; // r14d
  unsigned int *v31; // rax
  __int64 v32; // [rsp+8h] [rbp-88h]
  unsigned int v33; // [rsp+14h] [rbp-7Ch]
  __int64 v34; // [rsp+18h] [rbp-78h]
  unsigned __int64 v36; // [rsp+30h] [rbp-60h]
  unsigned __int16 *v37; // [rsp+38h] [rbp-58h]
  unsigned __int16 *v38; // [rsp+40h] [rbp-50h]
  __int64 v40; // [rsp+50h] [rbp-40h]
  int v41; // [rsp+58h] [rbp-38h]

  v37 = (unsigned __int16 *)(*(_QWORD *)(a3 + 8) - 40LL * (unsigned int)~*(_DWORD *)(a1 + 24));
  v33 = *((unsigned __int8 *)v37 + 4);
  v36 = *((unsigned __int8 *)v37 + 8) + (unsigned __int64)*((unsigned int *)v37 + 3) + 4 * (5LL * *v37 + 5);
  v5 = a1;
  while ( a2 )
  {
    v7 = *(_DWORD *)(a2 + 24);
    v8 = *(_QWORD *)(a2 + 40);
    v9 = *(_DWORD *)(a2 + 64);
    if ( v7 < 0 )
    {
      v10 = (unsigned __int16 *)(*(_QWORD *)(a3 + 8) - 40LL * (unsigned int)~v7);
      v11 = *((unsigned __int8 *)v10 + 9);
      v12 = v8 + 40LL * v9;
      if ( v12 != v8 )
      {
        v13 = *(_QWORD *)(a2 + 40);
        while ( *(_DWORD *)(*(_QWORD *)v13 + 24LL) != 10 )
        {
          v13 += 40;
          if ( v12 == v13 )
            goto LABEL_33;
        }
        v40 = *(_QWORD *)(*(_QWORD *)v13 + 96LL);
        if ( !*((_BYTE *)v10 + 9) && !v40 )
          goto LABEL_27;
        goto LABEL_9;
      }
LABEL_33:
      if ( *((_BYTE *)v10 + 9) )
      {
        v40 = 0;
LABEL_9:
        v14 = v33;
        v41 = *(_DWORD *)(v5 + 68);
        if ( v33 != v41 )
        {
          v32 = a2;
          v15 = 5LL * *v10 + 5;
          v16 = *((unsigned __int8 *)v10 + 8) + (unsigned __int64)*((unsigned int *)v10 + 3) + 4 * v15;
          v17 = &v10[v16];
          v18 = &v17[v11];
          v38 = v17;
          v19 = 0;
          v20 = v5;
          v21 = 0;
          v22 = v18;
          v23 = v20;
          do
          {
            v24 = *(_WORD *)(*(_QWORD *)(v23 + 48) + 16LL * v14);
            if ( v24 != 262 && v24 != 1 )
            {
              v25 = sub_33CF8A0(v23, v14, v16, v15, a5, v19);
              if ( (_BYTE)v25 )
              {
                v15 = v40;
                v26 = v37[v36 + v21];
                if ( v40 )
                {
                  v27 = *(_DWORD *)(v40 + 4 * ((unsigned __int64)(unsigned __int16)v26 >> 5));
                  if ( !_bittest(&v27, v26) )
                    return v25;
                }
                v28 = v38;
                if ( v22 != v38 )
                {
                  a5 = v26 - 1;
                  v34 = v23;
                  v29 = v25;
                  do
                  {
                    v16 = *v28;
                    if ( (_DWORD)v16 == v26
                      || v26 - 1 <= 0x3FFFFFFE
                      && (unsigned int)(v16 - 1) <= 0x3FFFFFFE
                      && (unsigned __int8)sub_E92070(a4, v26, v16) )
                    {
                      return v29;
                    }
                    ++v28;
                  }
                  while ( v22 != v28 );
                  v23 = v34;
                }
              }
            }
            ++v14;
            ++v21;
          }
          while ( v41 != v14 );
          v5 = v23;
          v8 = *(_QWORD *)(v32 + 40);
          v9 = *(_DWORD *)(v32 + 64);
        }
      }
    }
LABEL_27:
    if ( v9 )
    {
      v31 = (unsigned int *)(v8 + 40LL * (v9 - 1));
      a2 = *(_QWORD *)v31;
      if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v31 + 48LL) + 16LL * v31[2]) == 262 )
        continue;
    }
    return 0;
  }
  return 0;
}
