// Function: sub_7BC160
// Address: 0x7bc160
//
void __fastcall sub_7BC160(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD v9[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( *(_QWORD *)(a1 + 8) )
  {
    sub_7ADF70((__int64)v9, 0);
    sub_7AE360((__int64)v9);
    *(_QWORD *)v9[2] = qword_4F08560;
    v1 = qword_4F08540;
    qword_4F08560 = v9[1];
    if ( qword_4F08540 )
      qword_4F08540 = *(_QWORD *)qword_4F08540;
    else
      v1 = sub_823970(72);
    *(_QWORD *)v1 = 0;
    *(_QWORD *)(v1 + 8) = 0;
    *(_QWORD *)(v1 + 16) = 0;
    *(_QWORD *)(v1 + 24) = 0;
    sub_7ADF70(v1 + 32, 1);
    v2 = qword_4F08538;
    *(_QWORD *)(v1 + 24) = a1;
    *(_DWORD *)(v1 + 64) = 0;
    *(_QWORD *)v1 = v2;
    v3 = qword_4F08560;
    *(_BYTE *)(v1 + 68) = 0;
    *(_QWORD *)(v1 + 8) = v3;
    v4 = *(_QWORD *)(a1 + 8);
    qword_4F08538 = v1;
    *(_QWORD *)(v1 + 16) = v4;
    qword_4F08560 = 0;
    dword_4F061FC = 1;
    sub_7B8B50(v1 + 32, (unsigned int *)1, v5, v6, v7, v8);
  }
}
