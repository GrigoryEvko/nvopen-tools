// Function: sub_27C0DB0
// Address: 0x27c0db0
//
__int64 __fastcall sub_27C0DB0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // rsi
  unsigned __int8 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rcx
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v18; // rdi
  __int64 v19; // r8
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int8 v27; // [rsp+17h] [rbp-B9h]
  __int64 *v28; // [rsp+18h] [rbp-B8h]
  __int64 v29; // [rsp+28h] [rbp-A8h]
  __int64 *v30; // [rsp+30h] [rbp-A0h]
  __int64 v31; // [rsp+38h] [rbp-98h]
  __int64 v33; // [rsp+48h] [rbp-88h]
  __int64 *v34; // [rsp+50h] [rbp-80h] BYREF
  __int64 v35; // [rsp+58h] [rbp-78h]
  _BYTE v36[112]; // [rsp+60h] [rbp-70h] BYREF

  v34 = (__int64 *)v36;
  v35 = 0x800000000LL;
  sub_D474A0(a2, (__int64)&v34);
  v28 = &v34[(unsigned int)v35];
  if ( v34 != v28 )
  {
    v30 = v34;
    v27 = 0;
    while ( 1 )
    {
      v3 = sub_AA5930(*v30);
      v31 = v4;
      v5 = v3;
      if ( v3 != v4 )
        break;
LABEL_22:
      if ( v28 == ++v30 )
      {
        v28 = v34;
        goto LABEL_24;
      }
    }
    while ( (*(_DWORD *)(v5 + 4) & 0x7FFFFFF) == 0 )
    {
LABEL_18:
      v16 = *(_QWORD *)(v5 + 32);
      if ( !v16 )
        BUG();
      v5 = 0;
      if ( *(_BYTE *)(v16 - 24) == 84 )
        v5 = v16 - 24;
      if ( v31 == v5 )
        goto LABEL_22;
    }
    v6 = 0;
    v33 = 8LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
    while ( 1 )
    {
      v10 = *(_QWORD *)(*(_QWORD *)(v5 - 8) + 32LL * *(unsigned int *)(v5 + 72) + v6);
      if ( sub_D47930(a2) )
      {
        v11 = *(_QWORD *)(a1 + 16);
        v12 = sub_D47930(a2);
        if ( (unsigned __int8)sub_B19720(v11, v10, v12) )
        {
          v14 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v14 == v10 + 48 )
            goto LABEL_46;
          if ( !v14 )
            BUG();
          if ( (unsigned int)*(unsigned __int8 *)(v14 - 24) - 30 > 0xA )
LABEL_46:
            BUG();
          v15 = *(unsigned __int8 *)(v14 - 24);
          if ( (_BYTE)v15 == 31 )
          {
            v7 = *(_QWORD *)(v14 - 120);
LABEL_7:
            v8 = sub_D48480(a2, v7, v15, v13);
            if ( v8 )
            {
              v9 = *(_QWORD *)(*(_QWORD *)(v5 - 8) + 4 * v6);
              if ( *(_BYTE *)v9 == 84 && **(_QWORD **)(a2 + 32) == *(_QWORD *)(v9 + 40) )
              {
                v29 = *(_QWORD *)(*(_QWORD *)(v5 - 8) + 4 * v6);
                v18 = sub_D4B130(a2);
                if ( (*(_DWORD *)(v29 + 4) & 0x7FFFFFF) != 0 )
                {
                  v19 = *(_QWORD *)(v29 - 8);
                  v20 = 0;
                  while ( v18 != *(_QWORD *)(v19 + 32LL * *(unsigned int *)(v29 + 72) + 8 * v20) )
                  {
                    if ( (*(_DWORD *)(v29 + 4) & 0x7FFFFFF) == (_DWORD)++v20 )
                      goto LABEL_9;
                  }
                  v21 = *(_QWORD *)(v5 - 8) + 4 * v6;
                  v22 = *(_QWORD *)(v19 + 32 * v20);
                  v23 = *(_QWORD *)v21;
                  if ( v22 )
                  {
                    if ( v23 )
                    {
                      v24 = *(_QWORD *)(v21 + 8);
                      **(_QWORD **)(v21 + 16) = v24;
                      if ( v24 )
                        *(_QWORD *)(v24 + 16) = *(_QWORD *)(v21 + 16);
                    }
                    *(_QWORD *)v21 = v22;
                    v25 = *(_QWORD *)(v22 + 16);
                    *(_QWORD *)(v21 + 8) = v25;
                    if ( v25 )
                      *(_QWORD *)(v25 + 16) = v21 + 8;
                    *(_QWORD *)(v21 + 16) = v22 + 16;
                    *(_QWORD *)(v22 + 16) = v21;
                  }
                  else if ( v23 )
                  {
                    v26 = *(_QWORD *)(v21 + 8);
                    **(_QWORD **)(v21 + 16) = v26;
                    if ( v26 )
                      *(_QWORD *)(v26 + 16) = *(_QWORD *)(v21 + 16);
                    *(_QWORD *)v21 = 0;
                  }
                  sub_DAC8D0(*(_QWORD *)(a1 + 8), (_BYTE *)v5);
                  v27 = v8;
                }
              }
            }
            goto LABEL_9;
          }
          if ( (_BYTE)v15 == 32 )
          {
            v7 = **(_QWORD **)(v14 - 32);
            goto LABEL_7;
          }
        }
      }
LABEL_9:
      v6 += 8;
      if ( v33 == v6 )
        goto LABEL_18;
    }
  }
  v27 = 0;
LABEL_24:
  if ( v28 != (__int64 *)v36 )
    _libc_free((unsigned __int64)v28);
  return v27;
}
