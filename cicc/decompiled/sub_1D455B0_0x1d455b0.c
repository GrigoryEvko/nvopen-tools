// Function: sub_1D455B0
// Address: 0x1d455b0
//
void __fastcall sub_1D455B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v10; // rsi
  _QWORD *v11; // r8
  unsigned int v12; // edx
  __int64 *v13; // rcx
  __int64 v14; // rdi
  __int64 *v15; // rbx
  __int64 *v16; // r15
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rsi
  int v23; // ecx
  __int64 v24; // r10
  __int64 v25; // rax
  __int64 **v26; // rbx
  __int64 **v27; // r12
  __int64 *v28; // rsi
  int v29; // ecx
  __int64 v30; // [rsp-90h] [rbp-90h]
  int v31; // [rsp-84h] [rbp-84h]
  int v32; // [rsp-80h] [rbp-80h]
  __int64 v33; // [rsp-78h] [rbp-78h]
  __int64 v34; // [rsp-78h] [rbp-78h]
  _QWORD *v35; // [rsp-78h] [rbp-78h]
  _QWORD *v36; // [rsp-78h] [rbp-78h]
  __int64 v37; // [rsp-60h] [rbp-60h] BYREF
  __int64 **v38; // [rsp-58h] [rbp-58h] BYREF
  __int64 v39; // [rsp-50h] [rbp-50h]
  _BYTE v40[72]; // [rsp-48h] [rbp-48h] BYREF

  if ( (*(_BYTE *)(a2 + 26) & 1) != 0 )
  {
    v7 = *(_QWORD *)(a1 + 648);
    v38 = (__int64 **)v40;
    v39 = 0x200000000LL;
    v8 = *(unsigned int *)(v7 + 720);
    if ( (_DWORD)v8 )
    {
      v10 = *(_QWORD *)(v7 + 704);
      LODWORD(v11) = v8 - 1;
      v12 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (__int64 *)(v10 + 40LL * v12);
      v14 = *v13;
      if ( a2 == *v13 )
      {
LABEL_4:
        if ( v13 != (__int64 *)(v10 + 40 * v8) )
        {
          v15 = (__int64 *)v13[1];
          v16 = &v15[*((unsigned int *)v13 + 4)];
          if ( v16 != v15 )
          {
            do
            {
              v17 = *v15;
              if ( !*(_BYTE *)(*v15 + 49) && *(_WORD *)(a2 + 24) == 52 )
              {
                v18 = *(_QWORD *)(a2 + 32);
                v31 = *(_DWORD *)(v18 + 8);
                v30 = *(_QWORD *)v18;
                v33 = *(_QWORD *)(v18 + 40);
                if ( !sub_1D23600(a1, *(_QWORD *)v18) )
                {
                  if ( sub_1D23600(a1, v33) )
                  {
                    v19 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 88LL);
                    v20 = *(_QWORD **)(v19 + 24);
                    if ( *(_DWORD *)(v19 + 32) > 0x40u )
                      v20 = (_QWORD *)*v20;
                    v21 = sub_15C48E0(*(_QWORD **)(v17 + 24), 0, (__int64)v20, 0, 1);
                    v22 = *(_QWORD *)(v17 + 32);
                    v23 = *(_DWORD *)(v17 + 40);
                    v24 = v21;
                    v37 = v22;
                    if ( v22 )
                    {
                      v32 = v23;
                      v34 = v21;
                      sub_1623A60((__int64)&v37, v22, 2);
                      v23 = v32;
                      v24 = v34;
                    }
                    v11 = sub_1D24380(a1, *(_QWORD *)(v17 + 16), v24, v30, v31, *(_BYTE *)(v17 + 48), &v37, v23);
                    if ( v37 )
                    {
                      v35 = v11;
                      sub_161E7C0((__int64)&v37, v37);
                      v11 = v35;
                    }
                    v25 = (unsigned int)v39;
                    if ( (unsigned int)v39 >= HIDWORD(v39) )
                    {
                      v36 = v11;
                      sub_16CD150((__int64)&v38, v40, 0, 8, (int)v11, (int)a6);
                      v25 = (unsigned int)v39;
                      v11 = v36;
                    }
                    v38[v25] = v11;
                    *(_BYTE *)(v17 + 49) = 1;
                    LODWORD(v39) = v39 + 1;
                  }
                }
              }
              ++v15;
            }
            while ( v16 != v15 );
            v26 = v38;
            v27 = &v38[(unsigned int)v39];
            if ( v38 != v27 )
            {
              do
              {
                v28 = *v26++;
                sub_1D30360(a1, (__int64)v28, *v28, 0, (int)v11, a6);
              }
              while ( v27 != v26 );
              v27 = v38;
            }
            if ( v27 != (__int64 **)v40 )
              _libc_free((unsigned __int64)v27);
          }
        }
      }
      else
      {
        v29 = 1;
        while ( v14 != -8 )
        {
          a6 = (__int64 *)(unsigned int)(v29 + 1);
          v12 = (unsigned int)v11 & (v29 + v12);
          v13 = (__int64 *)(v10 + 40LL * v12);
          v14 = *v13;
          if ( a2 == *v13 )
            goto LABEL_4;
          v29 = (int)a6;
        }
      }
    }
  }
}
