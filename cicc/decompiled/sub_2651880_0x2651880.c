// Function: sub_2651880
// Address: 0x2651880
//
void __fastcall sub_2651880(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rax
  __int64 v11; // r10
  __int64 v12; // r11
  __int64 v13; // r15
  __int64 v14; // r13
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // r9
  __int64 v18; // r8
  __int64 v19; // rcx
  __int64 v20; // r12
  __int64 v21; // rbx
  __int64 v22; // r10
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int64 v30; // rdi
  __int64 v31; // [rsp+0h] [rbp-60h]
  __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  int v40; // [rsp+20h] [rbp-40h]
  __int64 v41[7]; // [rsp+28h] [rbp-38h] BYREF

  v41[0] = a6;
  if ( a4 && a5 )
  {
    if ( a4 + a5 == 2 )
    {
      if ( sub_2650F70(v41, a2, a1) )
      {
        v16 = *(_DWORD *)(a1 + 64);
        v17 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(a1 + 8) = 0;
        v18 = *(_QWORD *)(a1 + 16);
        v19 = *(_QWORD *)(a1 + 24);
        *(_QWORD *)(a1 + 16) = 0;
        v20 = *(_QWORD *)(a1 + 48);
        v21 = *(_QWORD *)(a1 + 56);
        *(_QWORD *)(a1 + 24) = 0;
        ++*(_QWORD *)(a1 + 40);
        v22 = *(_QWORD *)a1;
        *(_QWORD *)(a1 + 48) = 0;
        v23 = *(_QWORD *)(a1 + 32);
        *(_QWORD *)(a1 + 56) = 0;
        *(_DWORD *)(a1 + 64) = 0;
        v40 = v16;
        v31 = v22;
        *(_QWORD *)a1 = *(_QWORD *)a2;
        v33 = v17;
        *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
        v36 = v18;
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
        v38 = v19;
        *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24);
        *(_QWORD *)(a2 + 8) = 0;
        *(_QWORD *)(a2 + 16) = 0;
        *(_QWORD *)(a2 + 24) = 0;
        v24 = *(_QWORD *)(a2 + 32);
        v25 = *(unsigned int *)(a1 + 64);
        v26 = *(_QWORD *)(a1 + 48);
        *(_QWORD *)(a1 + 32) = v24;
        sub_C7D6A0(v26, 4 * v25, 4);
        ++*(_QWORD *)(a1 + 40);
        *(_QWORD *)(a1 + 56) = 0;
        *(_QWORD *)(a1 + 48) = 0;
        *(_DWORD *)(a1 + 64) = 0;
        v27 = *(_QWORD *)(a2 + 48);
        ++*(_QWORD *)(a2 + 40);
        v28 = *(_QWORD *)(a1 + 48);
        *(_QWORD *)(a1 + 48) = v27;
        LODWORD(v27) = *(_DWORD *)(a2 + 56);
        *(_QWORD *)(a2 + 48) = v28;
        LODWORD(v28) = *(_DWORD *)(a1 + 56);
        *(_DWORD *)(a1 + 56) = v27;
        LODWORD(v27) = *(_DWORD *)(a2 + 60);
        *(_DWORD *)(a2 + 56) = v28;
        LODWORD(v28) = *(_DWORD *)(a1 + 60);
        *(_DWORD *)(a1 + 60) = v27;
        LODWORD(v27) = *(_DWORD *)(a2 + 64);
        *(_DWORD *)(a2 + 60) = v28;
        v29 = *(unsigned int *)(a1 + 64);
        *(_DWORD *)(a1 + 64) = v27;
        v30 = *(_QWORD *)(a2 + 8);
        *(_DWORD *)(a2 + 64) = v29;
        *(_QWORD *)a2 = v31;
        *(_QWORD *)(a2 + 8) = v33;
        *(_QWORD *)(a2 + 16) = v36;
        *(_QWORD *)(a2 + 24) = v38;
        if ( v30 )
        {
          j_j___libc_free_0(v30);
          v29 = *(unsigned int *)(a2 + 64);
        }
        *(_QWORD *)(a2 + 32) = v23;
        sub_C7D6A0(*(_QWORD *)(a2 + 48), 4 * v29, 4);
        ++*(_QWORD *)(a2 + 40);
        *(_QWORD *)(a2 + 48) = v20;
        *(_QWORD *)(a2 + 56) = v21;
        *(_DWORD *)(a2 + 64) = v40;
        sub_C7D6A0(0, 0, 4);
      }
    }
    else
    {
      if ( a4 > a5 )
      {
        v13 = a4 / 2;
        v15 = sub_2651750(a2, a3, a1 + 72 * (a4 / 2), v41[0]);
        v12 = a1 + 72 * (a4 / 2);
        v11 = v15;
        v37 = 0x8E38E38E38E38E39LL * ((v15 - a2) >> 3);
      }
      else
      {
        v37 = a5 / 2;
        v34 = a2 + 72 * (a5 / 2);
        v10 = sub_2651630(a1, a2, v34, v41[0]);
        v11 = v34;
        v12 = v10;
        v13 = 0x8E38E38E38E38E39LL * ((v10 - a1) >> 3);
      }
      v32 = v11;
      v35 = v12;
      v14 = sub_2646940(v12, a2, v11);
      sub_2651880(a1, v35, v14, v13, v37, v41[0]);
      sub_2651880(v14, v32, a3, a4 - v13, a5 - v37, v41[0]);
    }
  }
}
