// Function: sub_88CDC0
// Address: 0x88cdc0
//
size_t sub_88CDC0()
{
  fwrite("\n/* Legacy configuration: <unnamed> */\n", 1u, 0x27u, qword_4F07510);
  return fwrite("#define LEGACY_TARGET_CONFIGURATION_NAME NULL\n", 1u, 0x2Eu, qword_4F07510);
}
