// Function: sub_34B45F0
// Address: 0x34b45f0
//
__int64 __fastcall sub_34B45F0(__int64 a1, unsigned int a2, int a3)
{
  __int64 v3; // rbx
  _QWORD *v4; // rax
  char *v5; // rax
  __int64 v6; // rdx
  __int16 *v7; // rcx
  int v8; // r12d
  __int16 *v9; // rcx
  __int16 v10; // si
  int v11; // r12d
  int v12; // r15d
  char *v13; // r12
  __int16 *v14; // rax
  char *v15; // rbx
  int v16; // eax
  unsigned __int16 v17; // r12
  __int64 result; // rax
  __int16 *v19; // rax
  __int16 *v20; // rdx
  __int64 v21; // rax
  __int16 *v22; // r12
  __int64 v23; // rsi
  unsigned int v24; // r13d
  __int64 v25; // [rsp+10h] [rbp-B0h]
  _QWORD *v26; // [rsp+18h] [rbp-A8h]
  __int64 v28; // [rsp+28h] [rbp-98h]
  __int64 v29; // [rsp+30h] [rbp-90h]
  unsigned __int16 *v30; // [rsp+38h] [rbp-88h]
  __int16 *v31; // [rsp+40h] [rbp-80h]
  __int16 v32; // [rsp+48h] [rbp-78h]
  int v33; // [rsp+48h] [rbp-78h]
  unsigned int v34[4]; // [rsp+4Ch] [rbp-74h] BYREF
  int v35; // [rsp+5Ch] [rbp-64h] BYREF
  int v36; // [rsp+60h] [rbp-60h] BYREF
  __int16 *v37; // [rsp+68h] [rbp-58h]
  __int16 v38; // [rsp+70h] [rbp-50h]
  int v39; // [rsp+78h] [rbp-48h]
  __int64 v40; // [rsp+80h] [rbp-40h]
  __int16 v41; // [rsp+88h] [rbp-38h]

  v3 = a1;
  v4 = *(_QWORD **)(a1 + 120);
  v34[0] = a2;
  v26 = v4;
  v5 = sub_E922F0(*(_QWORD **)(a1 + 32), a2);
  v30 = (unsigned __int16 *)&v5[2 * v6];
  if ( v5 == (char *)v30 )
  {
    v28 = *(_QWORD *)(a1 + 120);
    v29 = *(_QWORD *)(v28 + 104);
    v25 = v34[0];
LABEL_11:
    result = 4 * v25;
    if ( *(_DWORD *)(v29 + 4 * v25) == -1 || *(_DWORD *)(*(_QWORD *)(v28 + 128) + 4 * v25) != -1 )
    {
      *(_DWORD *)(v26[13] + 4 * v25) = a3;
      *(_DWORD *)(v26[16] + 4LL * v34[0]) = -1;
      sub_34B4520(v26 + 7, v34);
      sub_34B4190(*(_QWORD **)(v3 + 120), v34[0]);
      v19 = (__int16 *)(*(_QWORD *)(*(_QWORD *)(v3 + 32) + 56LL)
                      + 2LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v3 + 32) + 8LL) + 24LL * v34[0] + 4));
      v20 = v19 + 1;
      result = (unsigned int)*v19;
      v33 = v34[0] + result;
      if ( (_WORD)result )
      {
        v21 = (unsigned __int16)(LOWORD(v34[0]) + result);
        v22 = v20;
        while ( 1 )
        {
          v23 = *(_QWORD *)(v3 + 120);
          v24 = (unsigned __int16)v21;
          if ( *(_DWORD *)(*(_QWORD *)(v23 + 104) + 4 * v21) == -1
            || *(_DWORD *)(*(_QWORD *)(v23 + 128) + 4 * v21) != -1 )
          {
            *(_DWORD *)(v26[13] + 4 * v21) = a3;
            *(_DWORD *)(v26[16] + 4 * v21) = -1;
            v36 = (unsigned __int16)v21;
            sub_34B4520(v26 + 7, (unsigned int *)&v36);
            sub_34B4190(*(_QWORD **)(v3 + 120), v24);
          }
          result = (unsigned int)*v22++;
          if ( !(_WORD)result )
            break;
          v33 += result;
          v21 = (unsigned __int16)v33;
        }
      }
    }
  }
  else
  {
    v25 = v34[0];
    v7 = (__int16 *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 56LL)
                   + 2LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL) + 24LL * v34[0] + 8));
    v8 = *v7;
    v9 = v7 + 1;
    v28 = *(_QWORD *)(a1 + 120);
    v10 = v8;
    v11 = v34[0] + v8;
    v32 = v11;
    v12 = v11;
    v13 = v5;
    v14 = 0;
    v29 = *(_QWORD *)(v28 + 104);
    v15 = v13;
    if ( v10 )
      v14 = v9;
    v31 = v14;
    while ( 1 )
    {
      v16 = *(unsigned __int16 *)v15;
      v36 = v12;
      v41 = 0;
      v35 = v16;
      v17 = v16;
      v39 = 0;
      v37 = v31;
      v40 = 0;
      v38 = v32;
      if ( sub_2E46590(&v36, &v35) && *(_DWORD *)(v29 + 4LL * v17) != -1 )
      {
        result = *(_QWORD *)(v28 + 128);
        if ( *(_DWORD *)(result + 4LL * v17) == -1 )
          break;
      }
      v15 += 2;
      if ( v30 == (unsigned __int16 *)v15 )
      {
        v3 = a1;
        goto LABEL_11;
      }
    }
  }
  return result;
}
