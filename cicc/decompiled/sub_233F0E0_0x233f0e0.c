// Function: sub_233F0E0
// Address: 0x233f0e0
//
__int64 __fastcall sub_233F0E0(__int64 a1)
{
  __int64 v1; // rsi
  _QWORD *v2; // rbx
  unsigned __int64 v3; // r12
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // rax
  _QWORD *v10; // r14
  _QWORD *v11; // r15
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // r14
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  __int64 v16; // r13
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // rdi
  _QWORD *v19; // rsi
  _QWORD *v20; // r15
  __int64 v21; // rax
  unsigned __int64 v22; // r15
  unsigned __int64 v23; // r14
  unsigned __int64 v24; // rdi
  __int64 v25; // rdx
  _QWORD *v27; // rax
  _QWORD *v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rax
  _QWORD *v31; // r14
  _QWORD *v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rsi
  _QWORD *v35; // [rsp+10h] [rbp-B0h]
  _QWORD *v37; // [rsp+20h] [rbp-A0h]
  _QWORD *v38; // [rsp+20h] [rbp-A0h]
  _QWORD *v39; // [rsp+28h] [rbp-98h]
  _QWORD v40[5]; // [rsp+38h] [rbp-88h] BYREF
  void *v41; // [rsp+60h] [rbp-60h]
  _QWORD v42[11]; // [rsp+68h] [rbp-58h] BYREF

  v1 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD **)(a1 + 8);
    v39 = &v2[2 * v1];
    do
    {
      if ( *v2 != -8192 && *v2 != -4096 )
      {
        v3 = v2[1];
        if ( v3 )
        {
          sub_C7D6A0(*(_QWORD *)(v3 + 128), 16LL * *(unsigned int *)(v3 + 144), 8);
          v4 = *(_QWORD *)(v3 + 112);
          if ( v4 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 16LL))(v4);
          v5 = *(_QWORD *)(v3 + 48);
          if ( v5 != v3 + 64 )
            _libc_free(v5);
          v6 = *(_QWORD *)(v3 + 16);
          if ( v6 )
          {
            if ( *(_BYTE *)(v6 + 440) )
            {
              v34 = *(unsigned int *)(v6 + 416);
              *(_BYTE *)(v6 + 440) = 0;
              sub_C7D6A0(*(_QWORD *)(v6 + 400), 16 * v34, 8);
            }
            sub_C7D6A0(*(_QWORD *)(v6 + 368), 32LL * *(unsigned int *)(v6 + 384), 8);
            v7 = *(_QWORD *)(v6 + 240);
            if ( v7 != v6 + 256 )
              _libc_free(v7);
            v8 = *(_QWORD *)(v6 + 56);
            if ( v8 != v6 + 72 )
              _libc_free(v8);
            v9 = *(unsigned int *)(v6 + 48);
            if ( (_DWORD)v9 )
            {
              v10 = *(_QWORD **)(v6 + 32);
              v11 = &v10[4 * v9];
              do
              {
                if ( *v10 != -16 && *v10 != -4 )
                {
                  v12 = v10[1];
                  if ( v12 )
                    j_j___libc_free_0(v12);
                }
                v10 += 4;
              }
              while ( v11 != v10 );
              LODWORD(v9) = *(_DWORD *)(v6 + 48);
            }
            sub_C7D6A0(*(_QWORD *)(v6 + 32), 32LL * (unsigned int)v9, 8);
            j_j___libc_free_0(v6);
          }
          v13 = *(_QWORD *)(v3 + 8);
          if ( v13 )
          {
            v14 = *(_QWORD *)(v13 + 384);
            if ( v14 != v13 + 400 )
              _libc_free(v14);
            v15 = *(_QWORD *)(v13 + 296);
            if ( v15 != v13 + 312 )
              _libc_free(v15);
            v16 = *(_QWORD *)(v13 + 168);
            v17 = v16 + 48LL * *(unsigned int *)(v13 + 176);
            if ( v16 != v17 )
            {
              do
              {
                v17 -= 48LL;
                v18 = *(_QWORD *)(v17 + 16);
                if ( v18 != v17 + 32 )
                  _libc_free(v18);
              }
              while ( v16 != v17 );
              v17 = *(_QWORD *)(v13 + 168);
            }
            if ( v17 != v13 + 184 )
              _libc_free(v17);
            v19 = *(_QWORD **)(v13 + 8);
            v20 = &v19[9 * *(unsigned int *)(v13 + 16)];
            if ( v19 != v20 )
            {
              do
              {
                v21 = *(v20 - 7);
                v20 -= 9;
                if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
                  sub_BD60C0(v20);
              }
              while ( v19 != v20 );
              v20 = *(_QWORD **)(v13 + 8);
            }
            if ( v20 != (_QWORD *)(v13 + 24) )
              _libc_free((unsigned __int64)v20);
            j_j___libc_free_0(v13);
          }
          v22 = *(_QWORD *)v3;
          if ( *(_QWORD *)v3 )
          {
            v23 = *(_QWORD *)(v22 + 128);
            if ( v23 )
            {
              v24 = *(_QWORD *)(v23 + 40);
              if ( v24 != v23 + 56 )
                _libc_free(v24);
              j_j___libc_free_0(v23);
            }
            if ( *(_BYTE *)(v22 + 96) )
            {
              v30 = *(unsigned int *)(v22 + 88);
              *(_BYTE *)(v22 + 96) = 0;
              if ( (_DWORD)v30 )
              {
                v31 = *(_QWORD **)(v22 + 72);
                v32 = &v31[2 * v30];
                do
                {
                  if ( *v31 != -8192 && *v31 != -4096 )
                  {
                    v33 = v31[1];
                    if ( v33 )
                    {
                      v38 = v32;
                      sub_B91220((__int64)(v31 + 1), v33);
                      v32 = v38;
                    }
                  }
                  v31 += 2;
                }
                while ( v32 != v31 );
                LODWORD(v30) = *(_DWORD *)(v22 + 88);
              }
              sub_C7D6A0(*(_QWORD *)(v22 + 72), 16LL * (unsigned int)v30, 8);
            }
            v25 = *(unsigned int *)(v22 + 56);
            if ( (_DWORD)v25 )
            {
              v27 = *(_QWORD **)(v22 + 40);
              v40[0] = 2;
              v40[1] = 0;
              v40[2] = -4096;
              v28 = &v27[6 * v25];
              v40[3] = 0;
              v42[0] = 2;
              v42[1] = 0;
              v42[2] = -8192;
              v41 = &unk_49DDFA0;
              v42[3] = 0;
              do
              {
                v29 = v27[3];
                *v27 = &unk_49DB368;
                if ( v29 != -4096 && v29 != 0 && v29 != -8192 )
                {
                  v35 = v28;
                  v37 = v27;
                  sub_BD60C0(v27 + 1);
                  v28 = v35;
                  v27 = v37;
                }
                v27 += 6;
              }
              while ( v28 != v27 );
              v41 = &unk_49DB368;
              sub_D68D70(v42);
              sub_D68D70(v40);
              v25 = *(unsigned int *)(v22 + 56);
            }
            sub_C7D6A0(*(_QWORD *)(v22 + 40), 48 * v25, 8);
            sub_C7D6A0(*(_QWORD *)(v22 + 8), 24LL * *(unsigned int *)(v22 + 24), 8);
            j_j___libc_free_0(v22);
          }
          j_j___libc_free_0(v3);
        }
      }
      v2 += 2;
    }
    while ( v39 != v2 );
    v1 = *(unsigned int *)(a1 + 24);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 16 * v1, 8);
}
