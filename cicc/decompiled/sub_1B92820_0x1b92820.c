// Function: sub_1B92820
// Address: 0x1b92820
//
__int64 __fastcall sub_1B92820(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 *v4; // rdi
  __int64 *v5; // r13
  __int64 v6; // r14
  int v7; // eax
  __int64 *v8; // r10
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rcx
  _QWORD *v12; // rax
  __int64 v13; // rcx
  _QWORD *v14; // rdx
  __int64 *v16; // [rsp+8h] [rbp-48h]
  _QWORD *v17; // [rsp+10h] [rbp-40h]
  int v18; // [rsp+1Ch] [rbp-34h]

  if ( *(_BYTE *)(a2 + 16) == 54 )
    v4 = *(__int64 **)a2;
  else
    v4 = **(__int64 ***)(a2 - 48);
  v5 = sub_1B8E090(v4, a3);
  v6 = sub_13A4950(a2);
  v7 = sub_14A3620(*(_QWORD *)(a1 + 328));
  v8 = *(__int64 **)(a1 + 328);
  v9 = *(_QWORD *)(a1 + 320);
  v18 = v7;
  v10 = *(_QWORD *)(v9 + 504);
  if ( v10 == *(_QWORD *)(v9 + 496) )
    v11 = *(unsigned int *)(v9 + 516);
  else
    v11 = *(unsigned int *)(v9 + 512);
  v17 = (_QWORD *)(v10 + 8 * v11);
  v16 = v8;
  v12 = sub_15CC2D0(v9 + 488, a2);
  v13 = *(_QWORD *)(v9 + 504);
  if ( v13 == *(_QWORD *)(v9 + 496) )
    v14 = (_QWORD *)(v13 + 8LL * *(unsigned int *)(v9 + 516));
  else
    v14 = (_QWORD *)(v13 + 8LL * *(unsigned int *)(v9 + 512));
  for ( ; v14 != v12; ++v12 )
  {
    if ( *v12 < 0xFFFFFFFFFFFFFFFELL )
      break;
  }
  return v18
       + (unsigned int)sub_14A3500(v16, (unsigned int)*(unsigned __int8 *)(a2 + 16) - 24, (__int64)v5, v6, v12 != v17);
}
